# loss.py

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Callable, Optional
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

def string_similarity(s1, s2):
    return nltk.edit_distance(s1, s2) / max(len(s1), len(s2))


class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, 
                 reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None) -> None:
        super(CustomBCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)
    
    def set_posweight(self, new_pos_weight: torch.Tensor) -> None:
        """
        Set new positive weights for the loss function.
        
        Args:
            new_pos_weight (torch.Tensor): A tensor containing the new positive weights.
        """
        if new_pos_weight.size() != self.pos_weight.size():
            raise ValueError("New pos_weight must have the same size as the original pos_weight.")
        with torch.no_grad():
            self.pos_weight.copy_(new_pos_weight)

class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        return self.bce_loss(outputs, labels)
