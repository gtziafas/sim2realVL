from ..types import *

import torch
import torch.nn as nn 
import torch.nn.functional as F


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
    def __init__(self, input_dim: int, fusion_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.df = fusion_dim
        self.fc = nn.Linear(input_dim, self.df)
        self.rnn = nn.GRU(self.df, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.concat = ConcatFusion(fusion_dim)

    def forward(self, inps: Tuple[Tensor, ...]) -> Tensor:
        assert len(set([t.shape[0:-1] for t in inps])) == 1
        catted = self.concat(inps)
        out = self.fc(catted)
        out = F.gelu(out)
        out, _ = self.rnn(out)
        return out


class GRUSpatialFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.df = fusion_dim
        self.fc = nn.Linear(input_dim, self.df)
        self.rnn_x = nn.GRU(self.df, hidden_dim // 2, num_layers, bidirectional=False, batch_first=True)
        self.rnn_y = nn.GRU(self.df, hidden_dim // 2, num_layers, bidirectional=False, batch_first=True)
        self.concat = ConcatFusion(fusion_dim)

    @torch.no_grad()
    def sort(self, position: Tensor) -> Tuple[Tensor, Tensor]:
        _, indices_x = torch.sort(position[...,0], dim=-1)
        _, indices_y = torch.sort(position[...,1], dim=-1)
        return indices_x, indices_y       

    def forward(self, inps: Tuple[Tensor, ...]) -> Tensor:
        visual, position, phrase = inps
        xs = position[..., 0]

