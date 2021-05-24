from ..types import *
from sklearn.metrics import accuracy_score, hamming_loss

Metrics = Dict[str, Any]
MetricsFn = Callable[[array, array], Dict[str, Any]]


def vg_metrics(preds: Tensor, truths: List[array], ignore_idx: int = -1) -> Metrics:
    batch_size, num_boxes = preds.shape[0:2]
    correct_items = torch.ones_like(preds)
    correct_items[preds != truths]



    preds = array([p[t!=ignore_idx].argmax(-1) for p, t in zip(preds, truths)])
    truths = array([t[t!=ignore_idx].argmax(-1) for t in truths])
    batch_size, num_boxes = truths.shape[0:2]
    return {'accuracy': (preds == truths).sum() / (batch_size * num_boxes)}

def type_accuracy(preds: Tensor, truth: Tensor, ignore_idx: int = -1) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_items = torch.ones_like(predictions)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sents = correct_items.prod(dim=1)
    num_correct_sents = correct_sents.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truth[truth == ignore_idx])

    return ((predictions.shape[0], num_correct_sents),
            (predictions.shape[0] * predictions.shape[1] - num_masked_items, num_correct_items - num_masked_items))
