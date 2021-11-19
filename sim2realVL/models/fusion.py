from ..types import *

import torch
import torch.nn as nn 



class ConcatFusion(nn.Module):
    def __init__(self, fusion_dim: int):
        super().__init__()
        self.df = fusion_dim

    def forward(self, inps: Tuple[Tensor,...]) -> Tensor:
        assert len(set([t.shape[0:-1] for t in inps])) == 1
        return torch.cat(inps, dim=-1)


class Conv1x1Fusion(nn.Module):
    def __init__(self, fusion_dim: int, num_channels: int = 3):
        super().__init__()
        self.df = fusion_dim
        self.conv = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, inps: Tuple[Tensor, ...]) -> Tensor:
        assert len(set([t.shape for t in inps])) == 1
        stacked = torch.stack(inps, dim=-2)
        return self.conv(stacked).squeeze()


class GRUFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, num_layers: int = 1):
        super().__init__()
        self.d_inp = input_dim
        self.df = fusion_dim
        self.rnn = nn.GRU(self.d_inp, self.df, num_layers, bidirectional=False, batch_first=True)
        self.concat = ConcatFusion(fusion_dim)

    def forward(self, inps: Tuple[Tensor, ...]) -> Tensor:
        assert len(set([t.shape[0:-1] for t in inps])) == 1
        catted = self.concat(inps)
        out, _ = self.rnn(catted)
        return out