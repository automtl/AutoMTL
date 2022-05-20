from loguru import logger

from datasets.dataset import Dataset
from utils.utils import get_criterion

import torch
import torch.nn as nn

from layers.layers import MLP, TokenEmbedding

class AbstractMTLModel(nn.Module):
    """An abstract general framework for baseline MTL models.
    
    Args:
        dataset
        config
    """
    def __init__(self, dataset, config):
        super().__init__()
        
        # embedding layer and numerical feature preprocessing layer
        self.embedding_dim = config.embedding_dim
        self.task_num = config.task_num
        self.criterions = config.criterions
        self.task_weights = config.task_weights
        
        self.token_field_dims = dataset.field_token_dims
        self.token_field_seq_dims = dataset.field_token_seq_dims
        self.numerical_field_dim = len(dataset.float_fields)
        self.numerical_seq_field_dim = len(dataset.float_seq_fields)
        
        self.emb_field_num = (len(self.token_field_dims) + len(self.token_field_seq_dims) +
                          int(self.numerical_field_dim > 0) + int(self.numerical_seq_field_dim > 0))
        self.embed_out_dim = self.emb_field_num * self.embedding_dim
        # self.embed_out_dim = (len(self.token_field_dims) + len(self.token_field_seq_dims) +
        #                       int(self.numerical_field_dim > 0) + int(self.numerical_seq_field_dim > 0)) * self.embedding_dim
        
        self.token_embedding = None
        self.token_seq_embedding = None
        self.numerical_layer = None
        self.numerical_seq_layer = None
        
        if self.token_field_dims:
            self.token_embedding = TokenEmbedding(self.token_field_dims, self.embedding_dim)
        if self.token_field_seq_dims:
            self.token_seq_embedding = TokenEmbedding(self.token_field_seq_dims, self.embedding_dim)
            
        if self.numerical_field_dim > 0:
            # self.numerical_layer = nn.Linear(self.numerical_field_dim, self.embedding_dim, bias=False)
            self.numerical_layer = nn.Sequential(
                nn.Linear(self.numerical_field_dim, self.embedding_dim),
                # nn.BatchNorm1d(self.embedding_dim),
                # nn.ReLU()
            )
        if self.numerical_seq_field_dim > 0:
            # self.numerical_seq_layer = nn.Linear(self.numerical_seq_field_dim, self.embedding_dim, bias=False)
            self.numerical_seq_layer = nn.Sequential(
                nn.Linear(self.numerical_seq_field_dim, self.embedding_dim),
                # nn.BatchNorm1d(self.embedding_dim),
                # nn.ReLU()
            )
            
    def calculate_loss(self, preds, labels: torch.Tensor):
        """Compute loss for multi-task learning

        Args:
            preds (tensor): predictions
            labels (tenser): labels
            criterions (list(str)): types of loss functions for multi-task learning
        """
        losses = []
        # loss_items = []
        for i in range(self.task_num):
            criterion = get_criterion(self.criterions[i])
            # if self.criterions[i] in ['bce', 'cross_entropy']:
            #     loss = criterion(preds[i], labels[:, i].long())
            # else:
            #     loss = criterion(preds[i], labels[:, i])
            loss = criterion(preds[i], labels[:, i])
            # loss_items.append(loss.item())
            losses.append(self.task_weights[i] * loss)
        # return loss_items, sum(losses)
        return tuple(losses)
    
    def get_embedding_output(self, token_feature, float_feature, token_seq_feature, float_seq_feature):
        """Calculate embedding output for input layer.

        Args:
            token_feature (tensor): shape: [batch_size, token_field_num]
            float_feature (tensor): shape: [batch_size, float_field_num]
            token_seq_feature (tensor): shape [batch_size, token_seq_field_num, seq_len] 
            float_seq_feature (tensor): shape [batch_size, float_seq_field_num, seq_len]
        """
        embs = []
        if len(token_feature) > 0:
            embs.append(self.token_embedding(token_feature))
        if len(float_feature) > 0:
            embs.append(self.numerical_layer(float_feature))  # linear transformation to same dimension of embedding
        if len(token_seq_feature) > 0:
            embs.append(self.token_seq_embedding(token_seq_feature))
        if len(float_seq_feature) > 0:
            # NOTE: use mean of numerical sequence here, and linear transform to the same dimension of embedding
            # [B, F, L]
            mask = (float_seq_feature != 0)
            mask = mask.float()
            seq_len = torch.sum(mask, dim=-1)
            feat = torch.sum(float_seq_feature, dim=-1) / (seq_len + 1e-8)
            
            embs.append(self.numerical_seq_layer(feat))
            # embs.append(self.numerical_seq_layer(torch.mean(float_seq_feature, dim=-1)))
        
        emb = torch.cat(embs, dim=1).view(-1, self.embed_out_dim)
        return emb
