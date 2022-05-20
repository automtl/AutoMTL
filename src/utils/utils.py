from importlib.resources import path
import os
import shutil
import sys
from loguru import logger
import datetime
import numpy as np
import random
import math

from typing import Dict, List, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

PROJECT_PATH = os.path.abspath(os.path.join(__file__, '../../../'))
PROJECT_PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(PROJECT_PATH)))

logger.debug(f'current project path: {PROJECT_PATH}')

def get_latest_checkpoint(dir, tag=None):
    if not os.path.exists(dir):
        raise ValueError(f'{dir} not exists.')
    file_list = os.listdir(dir)
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    file = file_list[-1]
    iter = int(os.path.splitext(file)[0].split('_')[-1])

    return os.path.join(dir, file), iter

def get_current_context(key: str):
    return ContextStack.top(key)


def to_device(obj, device):
    """Move a tensor, tuple, list or dict onto device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError(f'{obj} has unsupported type {type(obj)}')


def set_seed(seed=666, cudnn_benchmark=True):
    """
    if cudnn_benchmark is True, performance might improve if the benchmarking feature is enabled,
    but cannot ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark


def set_logger(save_file_path=None):
    logger.remove()
    logger.add(sys.stdout)
    if save_file_path is None:
        return logger
    logger.add(save_file_path)

def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return ', '.join([str(metric) + ': ' + f'{value:.4f}' for metric, value in result_dict.items()])

def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y')
    # cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def ensure_dir(dir_path, force_removed=False):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (Str):
    """
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_file(file_path, force_removed=False):
    if force_removed:
        try:
            os.remove(file_path)
        except Exception as e:
            pass


def get_criterion(criterion_name):
    """Return criterion by name.

    Args:
        criterion_name (str)
    """
    if criterion_name == 'mse':
        return  nn.MSELoss()
    elif criterion_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f'Criterion {criterion_name} has not been implemented.')

def get_optimizer(optimizer_name, params, lr, momentum=0.9, weight_decay=0):
    """Return optimizer by name.

    Args:
        optimizer_name (str)
    """
    if optimizer_name == 'adagrad':
        return torch.optim.Adagrad(params, lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} has not been implemented.')

# def compute_loss(preds, labels, criterions):
#     """Compute loss for multi-task learning

#     Args:
#         preds (tensor): predictions
#         labels (tenser): labels
#         criterions (tenser): types of loss functions for multi-task learning
#     """

def get_tensorboard(log_dir, model_name, dataset_name):
    """Create a SummaryWriter of Tensorboard that can log Pytorch models and metrics.

    Args:
        board_dir (str): SummaryWriter's log_dir.
    Returns:
        SummeryWriter: it will write events and summaries to the event file.
    """
    base_path = 'log_tensorboard'
    dir_name = f'{model_name}_{dataset_name}_tensorboad'

    dir_path = os.path.join(PROJECT_PATH, log_dir, base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


class DecayScheduler:
    def __init__(self, base_weight=1.0, steps=15, decay_type='linear'):
        self.base_weight = base_weight
        self.cnt = 0
        self.decay_type = decay_type
        self.weight = base_weight
        self.steps = steps - 1
        
    def reset(self, base_weight=1.0, steps=15):
        self.base_weight = base_weight
        self.cnt = 0
        self.weight = base_weight
        self.steps = steps - 1
        # self.steps = steps

    def step(self):
        self.cnt += 1
        if self.decay_type == "cosine":
            self.weight = self.base_weight * (1 + math.cos(math.pi * self.cnt / self.steps)) / 2.0
        elif self.decay_type == "slow_cosine":
            self.weight = self.base_weight * math.cos((math.pi/2) * self.cnt / self.steps)
        elif self.decay_type == "linear":
            self.weight = self.base_weight * (self.steps - self.cnt) / self.steps
        else:
            self.weight = self.base_weight


class LossContainer:
    def __init__(self):
        self.sum = None
        self.avg = None
        self.cnt = 0
    def __str__(self) -> str:
        return f'{self.avg}'
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_desc_dict(self):
        if isinstance(self.avg, tuple):
            loss_desc = {f'train_loss{i + 1}': per_loss for i, per_loss in enumerate(self.avg)}
        else:
            loss_desc = {'train_loss': self.avg}
        return loss_desc
        
    def reset(self):
        self.sum = None
        self.avg = None
        self.cnt = 0
    def update(self, losses):
        self.cnt += 1
        if isinstance(losses, tuple):
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            self.sum = loss_tuple if self.sum is None else tuple(map(sum, zip(self.sum, loss_tuple)))
            self.avg = tuple(loss / self.cnt for loss in self.sum)
        elif isinstance(losses, torch.Tensor):
            self.sum = losses.item() if self.sum is None else self.sum + losses.item()
            self.avg = self.sum / self.cnt
        else:
            self.sum = losses if self.sum is None else self.sum + losses
            self.avg = self.sum / self.cnt


class EarlyStopper:
    def __init__(self, patience=5, method='min'):
        self._metrics = []
        self._patience = patience
        self._not_rise_steps = 0
        self._cur_max = None
        self._method = method

    @property
    def not_rise_steps(self):
        return self._not_rise_steps

    @not_rise_steps.setter
    def not_rise_steps(self, x):
        self._not_rise_steps = x

    @property
    def cur_max(self):
        return self._cur_max

    @cur_max.setter
    def cur_max(self, x):
        self._cur_max = x

    def add_metric(self, m):
        if self._method == 'min':
            m = -m
        self._metrics.append(m)
        if self._cur_max is None:
            self._cur_max = m
        if m > self._cur_max:
            self._cur_max = m
            self._not_rise_steps = 0
        else:
            self._not_rise_steps += 1

        if self._not_rise_steps >= self._patience:
            return True
        else:
            return False

class NoContextError(Exception):
    pass

class ContextStack:
    """This is to maintain a globally-accessible context environment that is visible to everywhere.

    Use ``with ContextStack(namespace, value):`` to initiate, and use ``get_current_context(namespace)`` to
    get the corresponding value in the namespace.

    Note that this is not multi-processing safe. Also, the values will get cleared for a new process.
    """
    _stack: Dict[str, List[Any]] = defaultdict(list)

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def __enter__(self):
        # for with ....
        self.push(self.key, self.value)
        return self

    def __exit__(self, *args, **kwargs):
        self.pop(self.key)

    @classmethod
    def push(cls, key, value):
        cls._stack[key].append(value)

    @classmethod
    def pop(cls, key):
        cls._stack[key].pop()

    @classmethod
    def top(cls, key):
        if not cls._stack[key]:
            raise NoContextError('Contex is empty.')
        return cls._stack[key][-1]