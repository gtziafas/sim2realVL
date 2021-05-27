from ..types import *
import torch
from torch.nn.utils.rnn import pad_sequence

Metrics = Dict[str, Any]
MetricsFn = Callable[[List[Tensor], List[Tensor], int], Metrics]


def vg_metrics(preds: List[Tensor], truths: List[Tensor], ignore_idx: int = -1) -> Metrics:
    preds = pad_sequence(preds, batch_first=True, padding_value=ignore_idx)
    truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx)

    num_samples, num_items = preds.shape[0:2]
    correct_items = torch.ones_like(preds)
    correct_items[preds != truths] = 0
    correct_items[truths == ignore_idx] = 1

    full_correct = correct_items.prod(dim=1)
    num_full_correct = full_corect.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truths[truths == ignore_idx])

    full_accuracy = num_full_correct / num_samples
    item_accuracy = (num_correct_items - num_masked_items) / (num_samples * num_items - num_masked_items)
    return {'accuracy': round(item_accuracy, 4), 'full_accuracy': round(full_accuracy, 4)}
