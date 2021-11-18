# from https://github.com/facebookresearch/pytorch3d/blob/main/projects/nerf/nerf/harmonic_embedding.py
from ..types import * 

import torch
import torch.nn as nn 


class MaskOut(nn.Module):
	def __init__(self)

class HarmonicEmbedder(nn.Module):
	def __init__(self,
		num_harmonics: int,
		omega0: float = 1.0,
		logspace: bool = True,
		include_input: bool = True):
		super().__init__()
		if logspace:
			frequencies = 2.0 ** torch.arange(num_harmonics, dtype=floatt)
		else:
			frequencies = torch.linspace(1.0, 2.0 ** (num_harmonics-1), dtype=floatt)

		self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
		self.include_input = include_input

	def forward(self, x: Tensor) -> Tensor:
		embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
		if self.include_input:
			return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
		else:
			return torch.cat((embed.sin(), embed.cos()), dim=-1)


def make_position_embedder(flag: int) -> nn.Module:
	if flag == 0:
		# no position embeddings at all 

	return HarmonicEmbedder()