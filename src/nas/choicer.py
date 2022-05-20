from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from utils.utils import NoContextError, get_current_context

def get_fixed_value(label: str):
    ret = get_current_context('fixed')
    try:
        return ret[label]
    except KeyError:
        raise KeyError(f'Fixed context with {label} not found. Existing values are {ret}')

class Mutable(nn.Module):
    """Implement trick for loading fixed architecture.
    """
    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__new__(cls)
        
        try:
            return cls.create_fixed_module(*args, **kwargs)
        except NoContextError:
            return super().__new__(cls)
        
    @classmethod
    def create_fixed_module(cls, *args, **kwargs):
        """Try to create a fixed module from fixed dict.
        """
        raise NotImplementedError

class LayerChoice(Mutable):
    """Layer choice selections one of the ``candidates``, then apply it on inputs and return results.
    
    Do not run foward in this module.
    
    Args:
        candidates: candidate modules
        prior (list(float)): prior distribution used in random sampling.
        label (str): identifier of layer choice
    """
    @classmethod
    def create_fixed_module(cls, candidates: Dict[str, nn.Module], label: str, **kwargs):
        chosen = get_fixed_value(label)
        if isinstance(candidates, list):
            result = candidates[int(chosen)]
        else:
            result = candidates[chosen]
            
        return result
    
    def __init__(self, candidates: Dict[str, nn.Module], label: str, *,
                 prior: Optional[List[float]] = None, ):
        super().__init__()
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = label
        
        self.names = []
        for name, module in candidates.items():
            self.add_module(name, module)
            self.names.append(name)
            
        self._first_module = self._modules[self.names[0]]
        
    @property
    def label(self):
        return self._label
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]
    
    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)
    
    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]
    
    def __len__(self):
        return len(self.names)
    
    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)
    
    def __repr__(self):
        return f'LayerChoice({self.candidates}, label={repr(self.label)})'
    
class ExpertChoice(Mutable):
    """Expert choice selections ``n_chosen`` outputs from ``choose_from`` (contians ``n_candidates`` keys).
    
    Do not run foward in this module.
    
    Args:
        n_chosen (int): Number of candidates to be chosen
        candidates: candidate modules
        prior (list(float)): prior distribution used in random sampling.
        label (str): identifier of layer choice
    """
    @classmethod
    def create_fixed_module(cls, candidates: Dict[str, nn.Module], n_chosen, label: str, **kwargs):
        chosen = get_fixed_value(label)
        return ChosenExperts(candidates, chosen, n_chosen)
    
    def __init__(self, candidates: Dict[str, nn.Module], n_chosen, label: str, *,
                 prior: Optional[List[float]] = None):
        super().__init__()
        self.n_chosen = n_chosen
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = label
        
        self.names = []
        for name, module in candidates.items():
            self.add_module(name, module)
            self.names.append(name)
            
        self._first_module = self._modules[self.names[0]]
        
    @property
    def label(self):
        return self._label
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]
    
    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)
    
    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]
    
    def __len__(self):
        return len(self.names)
    
    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)
    
    def __repr__(self):
        return f'ExpertChoice({self.candidates}, label={repr(self.label)})'
    
    
class ChosenExperts(nn.Module):
    """Choosing from experts and add corresponding gating netork.
    
    Args:
        candidates: candidats of mudules
        chosen
    """
    def __init__(self, candidates, chosen, n_chosen):
        super().__init__()
        self.chosen = chosen if isinstance(chosen, list) else [chosen]
        # self.n_chosen = len(chosen)
        self.n_chosen = n_chosen
        self.chosen = self.chosen[:n_chosen]
        
        self.experts = nn.ModuleList([candidates[i] for i in chosen])
        
        self.gate = nn.Sequential(
                    nn.Linear(self.experts[0].expert_in_dim, self.n_chosen, bias=False),
                    nn.Softmax(dim=1),
                )
        
    def forward(self, x):
        gate_value = self.gate(x).unsqueeze(1)
        experts_out = torch.stack([self.experts[i](x) for i in range(self.n_chosen)], dim=1)
        
        out = torch.bmm(gate_value, experts_out).squeeze(1)
        return out
