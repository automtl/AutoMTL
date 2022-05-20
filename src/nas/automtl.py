import os
import copy
from time import time
from typing import OrderedDict
import torch
import torch.nn as nn
from loguru import logger
from tqdm.auto import tqdm
from prefetch_generator import BackgroundGenerator

from datasets.dataset_utils import DataProvider
from utils.utils import DecayScheduler, EarlyStopper, LossContainer, dict2str, \
    ensure_dir, get_gpu_usage, to_device, PROJECT_PATH
from nas.utils import replace_layer_choice, replace_expert_choice
from trainer.trainer import AbstractTrainer
from nas.aux_losses import  JSDivLoss
from blocks.ops import DropPath

auxiliary_skip_weight = DecayScheduler()

class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice, drop_path_ratio=0.):
        super().__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(
            OrderedDict([(name, layer_choice[name]) for name in layer_choice.names])
        )
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
        
        self.drop_path = DropPath(drop_path_ratio)
        
        self.auxiliary_op = nn.Identity()


    def forward(self, *args, **kargs):
        op_results = torch.stack([self.drop_path(op(*args, **kargs)) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * torch.softmax(self.alpha, -1).view(*alpha_shape), dim=0)  \
            + auxiliary_skip_weight.weight * self.auxiliary_op(*args, **kargs)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        """Drop architecture params
        """
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]

class DartsExpertChoice(nn.Module):
    """use sigmoid to replace softmax here
    """
    def __init__(self, expert_choice):
        super().__init__()
        self.n_chosen = expert_choice.n_chosen
        self.name = expert_choice.label
        self.op_choices = nn.ModuleDict(
            OrderedDict([(name, expert_choice[name]) for name in expert_choice.names])
        )
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
        
        self.auxiliary_skip = nn.Identity()
        
    def forward(self, *args, **kargs):
        op_results = torch.stack([op(*args, **kargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * torch.softmax(self.alpha, -1).view(*alpha_shape), dim=0)
            

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        """Drop architecture params
        """
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        _, idx = torch.sort(self.alpha, descending=True)
        idx = idx.cpu().numpy()
        op_choices = list(self.op_choices.keys())
        return [op_choices[i] for i in idx]


class DartsTrainer(AbstractTrainer):
    """Darts trainer.
    """
    def __init__(self, config, model, train_loader, valid_loader):
        super().__init__(config, model)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.valid_data_provider = DataProvider(self.valid_loader)

        self.arch_learning_rate = config.arch_learning_rate
        self.arch_weight_decay = config.arch_weight_decay
        self.epochs = config.search_epochs
        self.clip_grad_norm = config.clip_grad_norm
        self.bactch_size = config.train_batch_size
        self.gpu_available = torch.cuda.is_available() and config.use_gpu
        self.device = 'cpu' if not self.gpu_available else config.device

        self.model = self.model.to(self.device)
        self.loss_func = self.model.calculate_loss
        self.unrolled = False if config.unrolled is None else config.unrolled
        
        self.drop_path_ratio = config.drop_path or 0.

        self.sim_loss = JSDivLoss(config.auxiliary_loss_weight, self.epochs - 5)
        
        auxiliary_skip_weight.reset(steps=self.epochs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            config.supernet_learning_rate,
            momentum=config.supernet_momentum,
            weight_decay=config.supernet_weight_decay
        )
        
        self.supernet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.epochs,
            eta_min=config.supernet_sgd_lr_min
        )

        self.start_epoch = 0
        self.search_time = 0
        self.top_archs = []
        self.top_k = config.top_k or 3

        self.checkpoint_dir = os.path.join(PROJECT_PATH, config.checkpoint_dir, f'{config.model}', self.config.dataset)
        ensure_dir(self.checkpoint_dir)
        saved_model_file = f'{self.config.dataset}-{self.config.model}-{config.expert_num}-{config.chosen_experts}.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.best_valid_loss = float('inf')
        self.early_stopper = EarlyStopper(patience=config.arch_early_stopping_step)

        self.expert_modules = []
        replace_expert_choice(self.model, DartsExpertChoice, self.expert_modules)
        
        self.nas_modules = []

        replace_layer_choice(self.model, lambda x: DartsLayerChoice(x, self.drop_path_ratio), self.nas_modules)

        for _, module in self.expert_modules:
            module = module.to(self.device)

        for _, module in self.nas_modules:
            module = module.to(self.device)
            
        self.expert_params = {}
        for _, m in self.expert_modules:
            self.expert_params[m.name] = m.alpha
        self.expert_optim = torch.optim.Adam(list(self.expert_params.values()), config.aggregation_learning_rate,
                                             betas=(0.5, 0.999), weight_decay=config.aggregation_weight_decay)

        # use the same archetecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), self.arch_learning_rate,
                                           betas=(0.5, 0.999), weight_decay=self.arch_weight_decay)

        need_research = config.need_research or False
        if not need_research:
            self._resume_checkpoint()

    def _save_checkpoint(self, epoch, **kwargs):
        """Store the model parameters information and training information.

        Args:
            epoch (int): current epoch id
            saved_model_file (str): save model file
        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'epoch': epoch,
            'search_time': self.search_time,
            'best_valid_loss': self.best_valid_loss,
            'early_stopper_cur_max': self.early_stopper.cur_max,
            'early_stopper_not_rise_steps': self.early_stopper.not_rise_steps,
            'supernet_state_dict': self.model.state_dict(),
            'expert_modules': self.expert_modules,
            'nas_modules': self.nas_modules,
            'top_archs': self.top_archs,
            
            'aux_loss_weight': self.sim_loss.weight,
            
            'optimizer': self.optimizer.state_dict(),
            'supernet_scheduler': self.supernet_scheduler.state_dict(),
            'expert_optimizer': self.expert_optim.state_dict(),
            'arch_optimizer': self.ctrl_optim.state_dict(),
        }
        torch.save(state, saved_model_file)
        logger.info(f'Saving current checkpoint (epoch: {epoch + 1}) to {saved_model_file}')


    def _resume_checkpoint(self, resume_file=None):
        """Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file
        """
        self.saved_model_file = resume_file or self.saved_model_file

        if not os.path.exists(self.saved_model_file):
            return

        checkpoint = torch.load(self.saved_model_file)
        
        self.sim_loss.weight = checkpoint['aux_loss_weight']

        self.start_epoch = checkpoint['epoch'] + 1
        auxiliary_skip_weight.cnt = self.start_epoch - 1
        auxiliary_skip_weight.step()
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.search_time = checkpoint['search_time']

        self.early_stopper.cur_max = checkpoint['early_stopper_cur_max']
        self.early_stopper.not_rise_steps = checkpoint['early_stopper_not_rise_steps']

        self.expert_modules = checkpoint['expert_modules']
        self.nas_modules = checkpoint['nas_modules']
        self.top_archs = checkpoint['top_archs']

        self.model.load_state_dict(checkpoint['supernet_state_dict'])
        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.supernet_scheduler.load_state_dict(checkpoint['supernet_scheduler'])
        self.expert_optim.load_state_dict(checkpoint['expert_optimizer'])
        self.ctrl_optim.load_state_dict(checkpoint['arch_optimizer'])

        logger.info(f'Checkpoint loaded. Resume training from epoch {self.start_epoch}.')


    def _train_epoch(self, epoch_idx, loss_func=None, tqdm_interval=20):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()

        if loss_func is not None:
            self.loss_func = loss_func

        iter_data = tqdm(
            enumerate(BackgroundGenerator(self.train_loader)),
            total=len(self.train_loader),
            desc=f'Train {epoch_idx + 1:>3}'
        )
        total_loss, valid_loss, train_loss = LossContainer(), LossContainer(), LossContainer()
        for batch_idx, train_data in iter_data:
            valid_data = self.valid_data_provider.next()
            train_data, valid_data = to_device(train_data, self.device), to_device(valid_data, self.device)

            # phase 1: architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                val_losses = self._unrolled_backward(train_data, valid_data)
            else:
                val_losses = self._backward(valid_data)
            valid_loss.update(val_losses)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.optimizer.zero_grad()
            if epoch_idx >= 5:
                self.expert_optim.zero_grad() # NOTE: optimize expert arch params
            _, losses, loss = self._preds_and_loss(train_data)
            
            total_loss.update(losses)
            train_loss.update(losses)

            if epoch_idx >= 5:
                sim_loss = self.sim_loss(list(self.expert_params.values()))
                loss = loss + sim_loss

            self._check_nan(loss)
            loss.backward()
            
            if self.clip_grad_norm:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                nn.utils.clip_grad.clip_grad_norm_(list(self.expert_params.values()), self.clip_grad_norm)
            self.optimizer.step()
            if epoch_idx >=5:
                self.expert_optim.step()

            if (batch_idx + 1) % tqdm_interval == 0:
                train_loss_desc = total_loss.to_desc_dict()
                # if self.gpu_available:
                #     iter_data.set_postfix(**train_loss_desc, GPU_RAM=get_gpu_usage(self.device))
                # else:
                iter_data.set_postfix(**train_loss_desc)
                total_loss.reset()
                # self._check_arch_params()
        if epoch_idx >= 5:
            self.sim_loss.step()  # NOTE: auxiliary loss weight decay
        return train_loss, valid_loss

    def _preds_and_loss(self, data):
        token_feature, float_feature, token_seq_feature, float_seq_feature, labels = data
        preds = self.model(token_feature, float_feature, token_seq_feature, float_seq_feature)
        losses = self.loss_func(preds, labels)
        if isinstance(losses, tuple):
            loss = sum(losses)
        else:
            loss = losses
        return preds, losses, loss

    def _backward(self, valid_data):
        """Simple backward with gradient descent.

        Using valid data to compute loss.
        """
        _, losses, loss = self._preds_and_loss(valid_data)
        loss.backward()
        return losses

    def _unrolled_backward(self, train_data, valid_data):
        """Compute unrolled loss and backward its gradients

        Using training data and valid data.
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]['lr']
        momentum = self.optimizer.param_groups[0]['momentum']
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self._compute_virtual_model(train_data, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        _, losses, loss = self._preds_and_loss(valid_data)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple([c.alpha for _, c in self.nas_modules])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, train_data)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)
        return losses

    def _compute_virtual_model(self, data, lr, momentum, weight_decay):
        """Compute unrolled weights w'
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, _, loss = self._preds_and_loss(data)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.optimizer.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, data):
        """Compute Hessian Matix

        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2 * eps)
        eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1e-8:
            logger.warning(f'In computing hessian, norm is smaller than 1E-8, cause eps to be {norm.item():.6f}.')

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps * dw`, w- = w - eps * dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            _, _, loss = self._preds_and_loss(data)
            dalphas.append(torch.autograd.grad(loss, [c.alpha for _, c in self.nas_modules]))

        dalpha_pos, dalpha_neg = dalphas  # dalpha {L_trn(w+)}, # dalpha {L_trn(w-)}
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


    def descretize(self, restore=False):
        for _, module in self.nas_modules:
            if isinstance(module, DartsLayerChoice):
                module.discretization = (not restore)
                if restore:
                    module.train()
                else:
                    module.eval()
        for _, module in self.expert_modules:
            module.discretization = (not restore)
            if restore:
                module.train()
            else:
                module.eval()

    def update_topk(self, arch, valid_loss):
        """update topk architecutres by valid loss
        """
        self.top_archs.append((valid_loss, arch))
        self.top_archs = sorted(self.top_archs)
        if len(self.top_archs) > self.top_k:
            self.top_archs = self.top_archs[:self.top_k]

    def _check_arch_params(self):
        for _, module in self.expert_modules:
            logger.debug(f'{module.name} alpha: {torch.softmax(module.alpha.data, dim=-1)}')
        for _, module in self.nas_modules:
                logger.debug(f'{module.name} alpha: {torch.softmax(module.alpha.data, dim=-1)}')
                break

    def fit(self):
        for epoch_idx in range(self.start_epoch, self.epochs):
            t = time()
            logger.info(f'epoch: {epoch_idx + 1}, supernet SGD lr: {self.supernet_scheduler.get_last_lr()}')
            train_loss, raw_val_loss = self._train_epoch(epoch_idx)
            self.supernet_scheduler.step()
            auxiliary_skip_weight.step()
            
            self._check_arch_params()

            cur_seach_time = time() - t
            self.search_time += cur_seach_time
            self._save_checkpoint(epoch_idx)

            logger.info(f'epoch={epoch_idx + 1}, training losses: {train_loss}, valid losses: {raw_val_loss}')
            logger.info(f'epoch={epoch_idx + 1}, arch: {self.export()}')

        logger.info(f'Finish training in {self.search_time}s.')

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        for name, module in self.expert_modules:
            if name not in result:
                result[name] = module.export()
        return result

    def export_topk(self):
        result = []
        for loss, arch in self.top_archs:
            result.append(arch)

        return result

