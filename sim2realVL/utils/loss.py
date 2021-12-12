from ..types import *
import torch.nn as nn 
import torch

class BCEWithLogitsIgnore(nn.Module):
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.core = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions[targets.ne(self.ignore_index)]
        targets = targets[targets.ne(self.ignore_index)]
        return self.core(predictions, targets)


class TripletHingeLoss(nn.Module):
    def __init__(self, ignore_index: int=-1, margin: float=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.margin = margin

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # preds: B x N, target: B x N
        batch_size = predictions.shape[0] 
        mask_positive = torch.where(targets == 1, predictions, torch.zeros_like(predictions))
        mask_negative = torch.where(targets == 0, predictions, torch.zeros_like(predictions))
        loss = torch.clamp(self.margin + mask_negative - mask_positive, 0)
        loss = loss.sum() / batch_size
        return loss