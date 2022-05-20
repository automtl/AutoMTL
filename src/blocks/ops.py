from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.layers import activate_layer

class DropPath(nn.Module):
    """Drop path with probability.
    
    Drop operations in block here.

    Attributes:
        p (float): probability of an path to be zeroed
    """
    def __init__(self, p=0.):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training and self.p > 0:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask
        return x


# Operators for one-shot NAS
class Block(nn.Module):
    """Abstract Block Module.
    
    Params in Setting assign by block inintilization, Params out setting are
    defined when blocks are defined, as hyper-parameters.
    
    Args:
        block_name (str): block name
        settings (dict): structure settings of current block
        use_bn: whether use batch norm
        dropout: dropout rate
    """
    def __init__(self, block_name, settings: dict):
        super().__init__()
        self.block_name = block_name
        self.block_in_dim = settings['block_in_dim']
        self.block_out_dim = settings['block_out_dim']
        
        self.dropout_rate = settings.get('dropout', 0)
        self.use_bn = settings.get('use_bn', True)
        
        if self.use_bn:
            self._batchnorm = nn.BatchNorm1d(self.block_out_dim)
        else:
            self._batchnorm = None
        self.dropout = nn.Dropout(self.dropout_rate)


class Conv1D(Block):
    """1D convolution layer.
    """
    def __init__(self, block_name, settings: dict, kernel_size, stride=1):
        super(Conv1D, self).__init__(block_name, settings)
        
        self.kernel = nn.parameter.Parameter(torch.randn(1, 1, kernel_size))
        self.padding = (kernel_size - 1) // 2
        
    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: [batch_size, in_dim]
        """
        x = inputs.unsqueeze(1)
        output = F.conv1d(x, self.kernel, padding=self.padding)
        output = output.squeeze(1)
        
        if self.use_bn:
            output = self._batchnorm(output)
        output = self.dropout(output)

        return output
        

class GatingLayer(Block):
    """Gating layer
    
    Element-wise enhanse fatures.
    """
    def __init__(self, block_name, settings: dict):
        super().__init__(block_name, settings)
        
        self.gate_layer = nn.Linear(self.block_in_dim, self.block_in_dim)
        
    def forward(self, inputs):
        """
        Args:
            inputs (tensor): shape: [batch_size, feature_dim]
        Returns:
            shape: [batch_size, feature_dim]
        """
        gates = self.gate_layer(inputs)  # [B, E]
        gates = torch.sigmoid(gates)
        
        out = inputs * gates
        
        if self.use_bn:
            out = self._batchnorm(out)
        out = self.dropout(out)
        
        return out


class MLP(Block):
    """
    This block applies MLP with one hidden layer.
    """
    def __init__(self, block_name, settings, hidden_size, activate='relu'):
        super(MLP, self).__init__(block_name, settings)
        
        self._hidden_size = hidden_size
        self._hidden_linear = nn.Linear(self.block_in_dim, hidden_size)
        self._hidden_batchnorm = nn.BatchNorm1d(hidden_size)
        
        self._output_linear = nn.Linear(hidden_size, self.block_out_dim)
        
        self._activate_func = activate_layer(activate)

    def forward(self, inputs):
        output = self._hidden_linear(inputs)
        # if self.use_bn:
        output = self._hidden_batchnorm(output)
        output = self._activate_func(output)
        output = self.dropout(output)

        output = self._output_linear(output)
        if self.use_bn:
            output = self._batchnorm(output)
        output = self._activate_func(output)
        output = self.dropout(output)

        return output


class CrossNet(Block):
    """
    This block applies CrossNet.
    """

    def __init__(self, block_name, settings, layer_num):
        super(CrossNet, self).__init__(block_name, settings)
        self._layer_num = layer_num

        self._w = nn.Parameter(torch.randn(self._layer_num, self.block_in_dim, 1))
        self._b = nn.Parameter(torch.randn(self._layer_num, self.block_in_dim))
        
        for i in range(layer_num):
            nn.init.xavier_uniform_(self._w[i])
        nn.init.zeros_(self._b)
        
    def forward(self, inputs):
        """
        :param inputs: [batch_size, nfields * embedding_dim]
        """
        x = x0 = inputs
        for i in range(self._layer_num):
            xw = torch.matmul(x, self._w[i])  # [B, 1]
            x = torch.mul(x0, xw) + self._b[i] + x  # [B, in_dim]

        output = x
        if self.use_bn:
            output = self._batchnorm(output)
        output = self.dropout(output)

        return output
        
    
class InnerProduct(Block):
    """Inner Product of IPNN
    """
    def __init__(self, block_name, settings):
        super(InnerProduct, self).__init__(block_name, settings)
        
        assert settings.get('num_fields', None) is not None, f'{block_name} show input num_fields'
        self._num_fields = settings.get('num_fields', None)
        
        assert settings.get('embedding_dim', None) is not None, f'{block_name} must input embedding_dim param'
        self.embedding_dim = settings.get('embedding_dim', None)
        
        self.W_z = nn.parameter.Parameter(torch.randn(self._num_fields, self._num_fields, self.embedding_dim))
        self.theta = nn.parameter.Parameter(torch.randn(self._num_fields * (self._num_fields - 1) // 2,
                                                        self._num_fields))
        
        self._output_affine = nn.Linear(self._num_fields + (self._num_fields * (self._num_fields - 1)) // 2,
                                        self.block_out_dim)
        
    def forward(self, inputs):
        """
        :params inputs: [batch_size, nfileds * embedding_size]
        """
        x = inputs.view([inputs.shape[0], self._num_fields, -1]) # [B, N, E]
        
        lz = torch.einsum('bnm,dnm->bd', x, self.W_z)  # (B, N)
        delta = torch.einsum('bnm,dn->bdnm', x, self.theta)  # (B, D1, N, M)
        lp = torch.einsum('bdnm,bdnm->bd', delta, delta)  # (B, D1)
        output = torch.cat((lz, lp), dim=1)
        
        output = self._output_affine(output)

        if self.use_bn:
            output = self._batchnorm(output)
        output = self.dropout(output)
        
        return output
    

class OuterProduct(Block):
    """Outer Product of OPNN
    """
    def __init__(self, block_name, settings):
        super(OuterProduct, self).__init__(block_name, settings)
        
        assert settings.get('num_fields', None) is not None, f'{block_name} show input num_fields'
        self._num_fields = settings.get('num_fields', None)
        
        assert settings.get('embedding_dim', None) is not None, f'{block_name} must input embedding_dim param'
        self.embedding_dim = settings.get('embedding_dim', None)
        
        self.W_z = nn.parameter.Parameter(torch.randn(self._num_fields, self._num_fields, self.embedding_dim))
        self.W_p = nn.parameter.Parameter(torch.randn(self._num_fields * (self._num_fields - 1) // 2,
                                          self._num_fields, self._num_fields))
        
        self._output_affine = nn.Linear(self._num_fields + (self._num_fields * (self._num_fields - 1)) // 2,
                                        self.block_out_dim)
        
    def forward(self, inputs):
        """
        :params inputs: [batch_size, nfileds * embedding_size]
        """
        x = inputs.view([inputs.shape[0], self._num_fields, -1]) # [B, N, E]
        
        lz = torch.einsum('bnm,dnm->bd', x, self.W_z)  # (B, N)
        
        x_sum = torch.sum(x, dim=-1)  # [B, N]
        p = torch.bmm(x_sum.unsqueeze(-1), x_sum.unsqueeze(1)) #[B, N, N]
        lp = torch.einsum('bmn,dmn->bd', p, self.W_p)  # (B, D1)
        
        output = torch.cat((lz, lp), dim=1)
        
        output = self._output_affine(output)

        if self.use_bn:
            output = self._batchnorm(output)
        output = self.dropout(output)
        
        return output
    

class SelfAttention(Block):
    """Multi-head Self-Attention Block.
    """
    def __init__(self, block_name, settings, num_heads, use_residual=True):
        super(SelfAttention, self).__init__(block_name, settings)
        
        assert settings.get('embedding_dim', None) is not None, f'{block_name} must input embedding_dim param'
        self.embedding_dim = settings.get('embedding_dim', None)
        
        assert self.embedding_dim % num_heads == 0, \
            f'attention_dim [{self.embedding_dim}] must be divide exactly by num_heads [{num_heads}]'
        
        self._num_heads = num_heads
        self.head_dim = self.embedding_dim // num_heads
        self._use_residual = use_residual
        
        # NOTE: Use one matrix to project all heads' query / key / value
        # This is not equive to use one matrix one head, but huggingface and pytorch all implement by this method. 
        self.linear_keys = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_values = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_query = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.final_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)


    def forward(self, inputs: torch.Tensor):
        """
        :params inputs: [batch_size, nfileds * embedding_size]
        """
        batch_size = inputs.size(0)
        head_dim = self.head_dim
        num_heads = self._num_heads
        x = inputs.contiguous().view([inputs.shape[0], -1, self.embedding_dim]) # [B, N, E]
        
        def shape(x):
            """projection"""
            return x.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_dim * num_heads)  # [B, N, E]
        
        # 1. project key, value, and query
        key = self.linear_keys(x)
        value = self.linear_values(x)
        query = self.linear_query(x)
        
        key, value, query = shape(key), shape(value), shape(query)  # [B, num_heads, N, head_dim]
        
        # 2. calculate and scale scores.
        query = query / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(2, 3))  # [B, num_heads, N, N]
        atten = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(atten, value)  # [B, num_heads, N, head_dim]
        output = unshape(output)
        output = self.final_linear(output)  # [B, N, E]
        
        if self._use_residual:
            output = output + x
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        output = output.contiguous().view([output.shape[0], -1])  # [B, nfields * embedding_size]
        
        return output

