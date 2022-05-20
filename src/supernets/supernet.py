import torch
import torch.nn as nn

from typing import OrderedDict

from layers.abstract import AbstractMTLModel
from layers.layers import MLP
from blocks.blocks import BLOCKS_POOL
from nas.choicer import LayerChoice, ExpertChoice 
from supernets.search_space_config import bottom_archs, expert_archs

class BottomModule(nn.Module):
    """Bottom Module for feature interaction.
    
    Args:
        layer_num
        block_type
    """
    def __init__(self, bottom_archs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(bottom_archs['layer_num']):
            settings = {
                'block_in_dim': bottom_archs['block_in_dim'],
                'block_out_dim': bottom_archs['block_out_dim'],
                'num_fields': bottom_archs['num_fields'],
                'embedding_dim': bottom_archs['embedding_dim']
            }
            self.layers.append(LayerChoice(OrderedDict([
                (block, BLOCKS_POOL[block](settings)) for block in bottom_archs['ops']
            ]), label=f'bottom_module_layer_{i}'))
            
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class ExpertModule(nn.Module):
    """Search space of Expert.
    """
    def __init__(self, expert_id, expert_archs):
        super().__init__()
        self.expert_in_dim = expert_archs['block_in_dim']
        self.layers = nn.ModuleList()
        for i in range(expert_archs['layer_num']):
            settings = {
                'block_in_dim': expert_archs['block_in_dim'],
                'block_out_dim': expert_archs['block_out_dim'],
            }
            self.layers.append(LayerChoice(OrderedDict([
                (block, BLOCKS_POOL[block](settings)) for block in expert_archs['ops']
            ]), label=f'expert_{expert_id}_layer_{i}'))
            
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

class Supernet(AbstractMTLModel):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        
        self.expert_num = config.expert_num
        
        bottom_archs['block_in_dim'] = bottom_archs['block_out_dim'] = self.embed_out_dim
        bottom_archs['num_fields'] = self.emb_field_num
        bottom_archs['embedding_dim'] = self.embedding_dim
        bottom_archs['dropout'] = config.dropout or 0.0
        bottom_archs['use_bn'] = config.use_bn or True
        
        expert_archs['layer_num'] = config.expert_layer_num
        expert_archs['dropout'] = config.dropout or 0.0
        expert_archs['use_bn'] = config.use_bn or True
        
        self.expert_in_projection = nn.Linear(self.embed_out_dim, expert_archs['block_in_dim'])
        
        self.bottom = BottomModule(bottom_archs)
        
        self.experts = OrderedDict([
            (f'expert_{i}', ExpertModule(i, expert_archs)) for i in range(self.expert_num)
        ])
        
        self.task_aggregators = nn.ModuleList([
            ExpertChoice(self.experts, n_chosen=config.chosen_experts, label=f'task_{i}_experts') \
                for i in range(self.task_num)
        ])
        
        self.towers = nn.ModuleList([
            MLP([expert_archs['block_out_dim']] + config.tower_layers, contain_output_layer=True) \
                for _ in range(self.task_num)
        ])
        
    def forward(self, token_feature, float_feature, token_seq_feature, float_seq_feature):
        emb = self.get_embedding_output(token_feature, float_feature, token_seq_feature, float_seq_feature)
        
        bottom_out = self.bottom(emb)
        
        bottom_out = self.expert_in_projection(bottom_out)
        
        outs = [self.towers[i](self.task_aggregators[i](bottom_out)).squeeze(1) for i in range(self.task_num)]
        
        return outs