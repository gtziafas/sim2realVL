from ..types import *
import torch

Metrics = Dict[str, Any]


def get_confusion_matrix(predictions: Tensor, truth: Tensor) -> Tensor:
    total_pos = truth[truth==1].shape[0]
    total_neg = truth[truth==0].shape[0]
    true_pos = predictions[truth==1].sum().item()
    true_neg = (~predictions[truth==0]).sum().item()
    false_pos = torch.where(torch.bitwise_and(predictions==1, truth==0), 1, 0).sum().item()
    false_neg = torch.where(torch.bitwise_and(predictions==0, truth==1), 1, 0).sum().item()
    return torch.tensor([[true_pos, true_neg], [false_pos, false_neg], [total_pos, total_neg]])


def get_metrics_from_matrix(matrix: Tensor) -> Metrics:
    TP, TN, FP, FN, P, N = matrix.flatten().tolist()
    return {"accuracy": round((TP+TN) / (P+N), 4),
            "true_positive_rate": round(TP / P, 4),
            "precision": round(TP / (TP+FP), 4),
            "recall":   round(TP / (TP+FN), 4),
            "f1_score": round(2*TP / (2*TP+FP+FN), 4)
            }