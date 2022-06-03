import torch 
import torch.nn as nn 
from torch.nn import functional as F

class Loss(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.log_softmax = F.log_softmax

    def forward(self, logits, targets): 
        return -torch.mean((self.log_softmax(logits, dim=1) * targets).sum(dim=1))