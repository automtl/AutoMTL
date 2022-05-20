import os

from collections import defaultdict

from ray import tune

import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from loguru import logger
from prefetch_generator import BackgroundGenerator

from utils.evaluate import calculate_eval_metrics
from utils.utils import get_gpu_usage, get_tensorboard, ensure_dir, get_local_time, EarlyStopper, dict2str, \
    PROJECT_PATH, to_device

class AbstractTrainer:
    """Trainer Class, used to manage the training and evaluation process.

    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        """Train the model based on train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, eval_func=None, is_test=False):
        r"""Evaluate the model based on the eval data.

        Use total loss as evaluate matric here.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            logger.info(message_output)

        self.model.eval()

        eval_func = eval_func or self.model.calculate_loss

        total_preds, total_labels = defaultdict(list), defaultdict(list)
        total_loss = None
        iter_data = enumerate(BackgroundGenerator(eval_data))
        for batch_idx, data in iter_data:
            token_feature, float_feature, token_seq_feature, float_seq_feature, labels = to_device(data, self.device)

            preds = self.model(token_feature, float_feature, token_seq_feature, float_seq_feature)

            losses = eval_func(preds, labels)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = loss.item() if total_loss is None else total_loss + loss.item()

            if is_test:
                for i in range(self.config.task_num):
                    if self.valid_metrics[i] in ['auc', 'acc']:
                        total_preds[i].extend(torch.sigmoid(preds[i]).tolist())
                        total_labels[i].extend(labels[:, i].long().tolist())
                    else:
                        total_preds[i].extend(preds[i].tolist())
                        total_labels[i].extend(labels[:, i].tolist())

        if isinstance(total_loss, tuple):
            total_loss = tuple(per_loss / len(eval_data) for per_loss in total_loss)
            result = {
                'total_loss': sum(total_loss)
            }
            result.update({'task_' + str(i): per_loss for i, per_loss in enumerate(total_loss)})
        else:
            total_loss = total_loss / len(eval_data)
            result = {
                'total_loss': total_loss
                }

        if is_test:
            result.update(calculate_eval_metrics(self.config.task_num, total_preds, total_labels, self.valid_metrics))

        return result


    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            optimizer_name (str, optional): The name of used optimizer. Defaults to ``self.optimizer_name``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        optimizer_name = kwargs.pop('optimizer_name', self.optimizer_name)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan.')


class Trainer(AbstractTrainer):
    """The basic trainer.

    Help train and evaluate model, resume cehckpoints...

    Args:
        config
        model
        early_stoper
    """
    def __init__(self, config, model):
        super().__init__(config, model)

        self.tensorboard = get_tensorboard(config.log_dir, config.model, config.dataset)
        self.optimizer_name = config.optimizer
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.epochs = config.max_epochs
        self.eval_step = min(config.eval_step, self.epochs)
        self.early_stopping_step = config.early_stopping_step
        self.clip_grad_norm = config.clip_grad_norm
        self.valid_metrics = config.valid_metrics
        self.gpu_available = torch.cuda.is_available() and config.use_gpu
        self.device = 'cpu' if not self.gpu_available else config.device
        
        if 'AutoMTL' in config.model:
            self.checkpoint_dir = os.path.join(PROJECT_PATH, config.checkpoint_dir, f'{config.model}', config.dataset,
                                            f'{config.expert_num}_{config.expert_layer_num}_{config.chosen_experts}_'
                                            f'{config.auxiliary_loss_weight}_'
                                            f'{config.dropout}_{config.drop_path}')
        else:
            self.checkpoint_dir = os.path.join(PROJECT_PATH, config.checkpoint_dir, 'ckpt',
                                            config.dataset, config.model)

        ensure_dir(self.checkpoint_dir)
        saved_model_file = f'{self.config.dataset}-{self.config.model}-{get_local_time()}.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_result = None
        self.early_stopper = EarlyStopper(patience=self.early_stopping_step)
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        self.model = self.model.to(self.device)

        self.hyper_tune = self.config.hyper_tune or False

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            optimizer_name (str, optional): The name of used optimizer. Defaults to ``self.optimizer_name``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        optimizer_name = kwargs.pop('optimizer_name', self.optimizer_name)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, tqdm_interval=10):
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
        loss_func = loss_func or self.model.calculate_loss

        iter_data = tqdm(
            enumerate(BackgroundGenerator(train_data)),
            total=len(train_data),
            desc=f'Train {epoch_idx:>3}'
        )
        total_loss, train_loss = None, None
        for batch_idx, data in iter_data:
            token_feature, float_feature, token_seq_feature, float_seq_feature, labels = to_device(data, self.device)

            preds = self.model(token_feature, float_feature, token_seq_feature, float_seq_feature)
            losses = loss_func(preds, labels)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                train_loss = loss_tuple if train_loss is None else tuple(map(sum, zip(train_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = loss.item() if total_loss is None else total_loss + loss.item()
                train_loss = loss.item() if train_loss is None else train_loss + loss.item()

            self._check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if (batch_idx + 1) % tqdm_interval == 0:
                if isinstance(total_loss, tuple):
                    train_loss_desc = {f'train_loss{i + 1}': per_loss / tqdm_interval for i, per_loss in enumerate(total_loss)}
                else:
                    train_loss_desc = {'train_loss': total_loss / tqdm_interval}
                # if self.gpu_available:
                #     iter_data.set_postfix(**train_loss_desc, GPU_RAM=get_gpu_usage(self.device))
                # else:
                iter_data.set_postfix(**train_loss_desc)
                total_loss = None

        train_loss = tuple(l / len(train_data) for l in train_loss) \
            if isinstance(train_loss, tuple) else train_loss / len(train_data)
        return train_loss

    def _valid_epoch(self, valid_data):
        """Valid model with valid data.

        Args:
            valid_data (DataLoader): validate data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result['total_loss']
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, **kwargs):
        """Store the model parameters information and training information.

        Args:
            epoch (int): current epoch id
            saved_model_file (str): save model file
        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'early_stopper_cur_max': self.early_stopper.cur_max,
            'early_stopper_not_rise_steps': self.early_stopper.not_rise_steps,
            'best_valid_loss': self.best_valid_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, saved_model_file)
        logger.info(f'Saving current checkpoint to {saved_model_file}')

    def resume_checkpoint(self, resume_file):
        """Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file
        """
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_valid_loss = checkpoint['best_valid_loss']

        self.early_stopper.cur_max = checkpoint['early_stopper_cur_max']
        self.early_stopper.not_rise_steps = checkpoint['early_stopper_not_rise_steps']

        # load architecture params from checkpoint
        if checkpoint['config'].model != self.config.model:
            logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f'Checkpoint loaded. Resume training from epoch {self.start_epoch}.')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan.')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = f'epoch {epoch_idx} [training time {e_time - s_time:.2f}s, '

        if isinstance(losses, tuple):
            train_loss_output += f'total_loss: {sum(losses):.4f}, '
            train_loss_output += ', '.join(f'task_{idx}: {loss:.4f}' for idx, loss in enumerate(losses))
        else:
            train_loss_output += f'loss: {losses:.4f}'
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + 'task_' + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)


    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'model': self.config.model,
            'dataset': self.config.dataset,
            'device': self.config.device,
            'optimizer': self.config.optimizer_name,
            'learning_rate': self.config.learning_rate,
            'train_batch_size': self.config.train_batch_size
        }
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})


    def fit(self, train_data, valid_data=None, saved=True, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            saved (bool, optional): whether to save the model parameters, default: True
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        logger.info(f'Start to train model {self.config.model}')
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time.time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time.time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time.time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                is_stop = self.early_stopper.add_metric(valid_score)
                valid_end_time = time.time()

                valid_result_output = f'epoch {epoch_idx} [evaluating time: {valid_end_time - valid_start_time:.2f}s, ' \
                    f'valid result - {dict2str(valid_result)}]'

                logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if valid_score < self.best_valid_loss:
                    self.best_valid_loss = valid_score
                    if saved:
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if self.hyper_tune:
                    with tune.checkpoint_dir(epoch_idx) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, f'{self.config.model}_checkpoint')
                        self._save_checkpoint(epoch_idx, saved_model_file=path)
                    tune.report(loss=valid_score)

                if is_stop:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.early_stopper.not_rise_steps * self.eval_step)
                    logger.info(stop_output)
                    break

        self._add_hparam_to_tensorboard(self.best_valid_loss)
        return self.best_valid_loss, self.best_valid_result

