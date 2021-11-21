from ..types import * 

import torch
import torch.nn as nn 


class PositionEmbedder(nn.Module):
	''' Converts a (x,y,w,h) bounding box from
		cv2-like coordinates 
			top-left 		<-> (0,0)
			bottom-right	<-> (W,H)
		into normalized range
			top-left		<-> (-1,1)
			bottom-right	<-> (1,1)
		while also returning intermediate features
			[x, y, x+w/2, y+h/2, x+w, y+h, w, h]
	'''
	def __init__(self, 
		embedder: nn.Module,
		img_height: int = 480,
		img_width: int = 640
	):
		super().__init__()
		self.W, self.H = img_width, img_height
		self.emb = embedder
		self.num_features = self.emb.num_features

	def x_t(self, x: Tensor) -> Tensor:
		return x * 2 / self.W - 1

	def y_t(self, y: Tensor) -> Tensor:
		return y * 2 / self.H - 1

	def forward(self, x: Tensor) -> Tensor:
		box = torch.empty(*x.shape[0:2], 8).to(x.device)
		box[...,0] = self.x_t(x[...,0])
		box[...,1] = self.y_t(x[...,1])
		box[...,2] = self.x_t(x[...,0] + x[...,2] / 2)
		box[...,3] = self.y_t(x[...,1] + x[...,3] / 2)
		box[...,4] = self.x_t(x[...,0] + x[...,2])
		box[...,5] = self.y_t(x[...,1] + x[...,3])
		box[...,6] = x[...,2] / self.W 
		box[...,7] = x[...,3] / self.H
		# box = torch.cat((
		# 		x[...,0] * 2 / self.W - 1,
		# 		x[...,1] * 2 / self.H - 1,
		# 		(x[...,0] + x[...,2] / 2) * 2 / self.W - 1,
		# 		(x[...,1] + x[...,3] / 2) * 2 / self.H - 1,
		# 		(x[...,0] + x[...,2]) * 2 / self.W - 1,
		# 		(x[...,1] + x[...,3]) * 2 / self.H - 1,
		# 		x[...,2] / self.W,
		# 		x[...,3] / self.H)
		# ).reshape(batch_size, 8)

		return self.emb(box)


class MaskOut(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_features = 1

	def forward(self, x: Tensor) -> Tensor:
		return torch.zeros((*x.shape[0:-1], 1), device=x.device)


class Identity(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_features = 8

	def forward(self, x: Tensor) -> Tensor:
		return x 


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
		self.num_features = 8 * 2*num_harmonics + 8 if include_input else 8 * 2*num_harmonics

	def forward(self, x: Tensor) -> Tensor:
		embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
		if self.include_input:
			return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
		else:
			return torch.cat((embed.sin(), embed.cos()), dim=-1)


def make_position_embedder(flag: str) -> nn.Module:
	if flag == 'no':
		# mask out position
		return PositionEmbedder(embedder=MaskOut())

	elif flag == 'raw':
		# keep raw positions
		return PositionEmbedder(embedder=Identity())

	elif flag == 'harmonic':
		# apply harmonic transforms to each position feature
		return PositionEmbedder(embedder=HarmonicEmbedder(num_harmonics=16, include_input=False))

	else:
		raise ValueError("Check models.position_embedder for valid options")