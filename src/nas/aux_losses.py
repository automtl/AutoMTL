import torch
import math
import torch.nn as nn
import torch.nn.functional as F
    
class JSDivLoss(nn.Module):
    """Jensen-Shannon divergence of expert arch params.

    Encourage them to be different.
    """
    def __init__(self, init_weight, max_epochs=None):
        super().__init__()
        self.weight = init_weight
        if max_epochs is not None:
            self.weight_step = init_weight / max_epochs
        
    def step(self):
        self.weight -= self.weight_step
        
    def entropy(self, x):
        return torch.dot(x, torch.log2(x))

    def forward(self, alphas):
        n = len(alphas)
        probs = []
        for i in range(n):
            probs.append(torch.softmax(alphas[i], dim=-1))
        means = sum(probs) / n
        
        loss = self.entropy(means)
        for prob in probs:
            loss = loss - self.entropy(prob) / n
        
        return self.weight * loss