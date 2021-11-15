from ..types import *
import torch

Metrics = Dict[str, Any]

def multi_class_vg_metrics(predictions: Tensor, truth: Tensor, ignore_idx: int = -1) -> Metrics:
    correct_items = torch.ones_like(predictions)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sents = correct_items.prod(dim=1)
    num_correct_sents = correct_sents.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truth[truth == ignore_idx])

    return ((predictions.shape[0], num_correct_sents),
            (predictions.shape[0] * predictions.shape[1] - num_masked_items, num_correct_items - num_masked_items))


def vg_metrics(predictions: Tensor, truth: Tensor) -> Tuple[int, int]:
    predictions = predictions[truth == 1]
    truth = truth[truth == 1]
    num_correct = (predictions == truth).sum().item()
    return num_correct, predictions.shape[0]