import torch
import torch.nn as nn

from blocks.ops import FM, MLP, Conv1D, CrossNet, InnerProduct, OuterProduct, SENETLayer, SelfAttention, GatingLayer


# blocks pool to use
BLOCKS_POOL = {
    'skipconnect':
        lambda _: nn.Identity(),
    'GatingLayer':
        lambda settings: GatingLayer('GatingLayer', settings),
        
    'InnerProduct':
        lambda settings: InnerProduct('InnerProduct', settings),
    'OuterProduct':
        lambda settings: OuterProduct('OuterProduct', settings),

    'MLP-16':
        lambda settings: MLP('MLP-16', settings, 16),
    'MLP-32':
        lambda settings: MLP('MLP-32', settings, 32),
    'MLP-64':
        lambda settings: MLP('MLP-64', settings, 64),
    'MLP-128':
        lambda settings: MLP('MLP-128', settings, 128),
    'MLP-256':
        lambda settings: MLP('MLP-256', settings, 256),
    'MLP-512':
        lambda settings: MLP('MLP-512', settings, 512),
    'MLP-1024':
        lambda settings: MLP('MLP-1024', settings, 1024),

    'Crossnet-1':
        lambda settings: CrossNet('Crossnet-1', settings, 1),
    'Crossnet-2':
        lambda settings: CrossNet('Crossnet-2', settings, 2),
    'Crossnet-3':
        lambda settings: CrossNet('Crossnet-3', settings, 3),
    'Crossnet-4':
        lambda settings: CrossNet('Crossnet-4', settings, 4),
    'Crossnet-5':
        lambda settings: CrossNet('Crossnet-5', settings, 5),
    'Crossnet-6':
        lambda settings: CrossNet('Crossnet-6', settings, 6),

    'SelfAttention-1':
        lambda settings: SelfAttention('SelfAttention-1', settings, 1),
    'SelfAttention-2':
        lambda settings: SelfAttention('SelfAttention-2', settings, 2),
    # 'SelfAttention-3':
    #     lambda settings: SelfAttention('SelfAttention-3', settings, 3),
    'SelfAttention-4':
        lambda settings: SelfAttention('SelfAttention-4', settings, 4),
    'SelfAttention-8':
        lambda settings: SelfAttention('SelfAttention-4', settings, 8),
        
    'Conv1D-3':
        lambda settings: Conv1D('Conv1D-3', settings, 3),
    'Conv1D-5':
        lambda settings: Conv1D('Conv1D-3', settings, 5),
    'Conv1D-7':
        lambda settings: Conv1D('Conv1D-3', settings, 7),
}