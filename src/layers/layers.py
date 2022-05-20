import torch
import torch.nn as nn

from datasets.dataset import Dataset

def activate_layer(activate_name='relu', emb_dim=None):
    """Construct activation layer.

    Args:
        activate_name (str): name of activation function. Defaults to 'relu'.
        emb_dim (int, optional): used for Dice activation. Defaults to None.

    Returns:
        activation: activation layer
    """
    if activate_name is None:
        return None
    if activate_name == 'sigmoid':
        activation = nn.Sigmoid()
    elif activate_name == 'tanh':
        activation = nn.Tanh()
    elif activate_name == 'relu':
        activation = nn.ReLU()
    elif activate_name == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif activate_name == 'none':
        activate_name = None
    else:
        raise NotImplementedError(f'activation function {activate_name} is not implemented.')

    return activation

class Embedding(nn.Module):
    r""" Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 2D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """
    def __init__(self, field_dims, offsets, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = offsets

    def forward(self, input_x: torch.Tensor):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output

class TokenEmbedding(nn.Module):
    r""" Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 2D tensor with shape:``(batch_size,field_size)`` or 3D tensor.

    Return:
        output: list(tensor),  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)`` or 4D tensor.
    """
    def __init__(self, field_dims, embedding_dim):
        super(TokenEmbedding, self).__init__()

        self.field_dims = field_dims

        self.embedding_tables = nn.ModuleList()

        for dim in self.field_dims:
            embedding = nn.Embedding(dim, embedding_dim)
            nn.init.xavier_uniform_(embedding.weight.data)
            self.embedding_tables.append(embedding)
    def forward(self, input_x: torch.Tensor):
        embeddings = []
        num_fields = input_x.shape[1]
        is_seq = len(input_x.shape) > 2
        for i in range(num_fields):
            if is_seq:
                x = input_x[:, i]
                mask = (x != 0)
                mask = mask.float()
                mask.unsqueeze_(-1)
                seq_lens = torch.sum(mask, dim=1)
                embedding = self.embedding_tables[i](input_x[:, i])
                embedding = embedding * mask
                embedding = torch.sum(embedding, dim=1) / (seq_lens + 1e-8)
            else:
                embedding = self.embedding_tables[i](input_x[:, i])
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=1)

        return embeddings

class MLP(nn.Module):
    r"""MLP Layers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """
    def __init__(self, layers, dropout=0., activate='relu', bn=True, init_method=None,
                 contain_output_layer=False):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activate = activate
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for i, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activate_func = activate_layer(activate_name=activate)
            if activate_func is not None:
                mlp_modules.append(activate_func)
            mlp_modules.append(nn.Dropout(p=self.dropout))

        if contain_output_layer:
            mlp_modules.append(nn.Linear(self.layers[-1], 1))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.)

    def forward(self, input_x):
        return self.mlp_layers(input_x)