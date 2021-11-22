from ..types import *

import torch
import torch.nn as nn 


class MLP(nn.Module):
  def __init__(self, hidden_sizes: List[int],
                     dropout_rates: List[float],
                     num_classes: int,
                     input_dim: int,
                     activation_fn: nn.Module = nn.GELU):
      super().__init__()
      assert len(hidden_sizes) == len(dropout_rates)
      self.activation_fn = activation_fn
      self.num_hidden_layers = len(hidden_sizes)
      self.layers = nn.ModuleList([self.layer(input_dim, hidden_sizes[0], dropout_rates[0])])
      self.layers.extend([self.layer(hidden_sizes[i], hidden_sizes[i+1], dropout_rates[i+1]) 
          for i in range(self.num_hidden_layers-1)])
      self.layers = nn.Sequential(*[*self.layers, nn.Linear(hidden_sizes[-1], num_classes)])

  def layer(self, inp_dim: int, out_dim: int, dropout: float) -> nn.Module:
      return nn.Sequential(nn.Linear(inp_dim, out_dim),
                            self.activation_fn(),
                            nn.Dropout(dropout))

  def forward(self, x: Tensor) -> Tensor:
      return self.layers(x)


class GRUContext(nn.Module):
  def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int, bidirectional: bool = False):
      super().__init__()
      self.bidirectional = bidirectional
      self.gru = nn.GRU(inp_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)

  def forward(self, x: Tensor) -> Tensor:
      dh = self.gru.hidden_size
      H, _ = self.gru(x)  # B x T x (2*)D
      out = H[:, -1, :] if not self.bidirectional else torch.cat((H[:, -1, :dh], H[:, 0, dh:]), dim=-1)
      return out


class CNNFeatures(nn.Module):
  def __init__(self, num_blocks: int, dropout_rates: List[float], 
               conv_kernels: List[int], pool_kernels: List[int],
               input_channels: int = 3, activation_fn: nn.Module = nn.GELU):
      super().__init__()
      assert num_blocks == len(conv_kernels) == len(pool_kernels) == len(dropout_rates)
      self.activation_fn = activation_fn
      self.blocks = nn.Sequential(self.block(input_channels, 16, conv_kernels[0], pool_kernels[0], dropout_rates[0]),
                                  *[self.block(2**(3+i), 2**(4+i), conv_kernels[i], pool_kernels[i], dropout_rates[i])
                                  for i in range(1, num_blocks)])

  def block(self, 
            in_channels: int, 
            out_channels: int, 
            conv_kernel: int, 
            pool_kernel: int, 
            dropout: float = 0., 
            conv_stride: int = 1
           ):
      return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                           self.activation_fn(),
                           nn.MaxPool2d(kernel_size=pool_kernel),
                           nn.Dropout(p=dropout)
                          ) 

  def forward(self, x: Tensor) -> Tensor:
      return self.blocks(x)


class CNNClassifier(nn.Module):
    def __init__(self, feature_extractor: CNNFeatures, head: MLP):
        super().__init__()
        self.features = feature_extractor
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        # x: B x 3 x H x W
        x = self.features(x).flatten(1) # B x D
        return self.head(x)


class AttentionLayer(nn.Module):
  def __init__(self, 
               hidden_dim: int,
               similarity: nn.Module = nn.Softmax(dim=-1)
              ):
    super().__init__()
    self.attn_vector = nn.Parameter(torch.rand(hidden_dim))
    self.similarity = similarity

  def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    prod = x @ self.attn_vector
    scores = self.similarity(prod)
    out = scores.unsqueeze(-1) * x
    context = out.mean(dim=1)
    return out, context


class GRUAttentionContext(nn.Module):
  def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int, bidirectional: bool = False):
    super().__init__()
    self.bidirectional = bidirectional
    self.gru = nn.GRU(inp_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
    self.attn = AttentionLayer(hidden_dim=hidden_dim if not bidirectional else 2 * hidden_dim)

  def forward(self, x: Tensor) -> Tensor:
    dh = self.gru.hidden_size
    H, _ = self.gru(x)  # B x T x (2*)D
    H, context = self.attn(H)
    return context


class BilinearAttentionLayer(nn.Module):
  def __init__(self, 
               hidden_dim: int,
               similarity: nn.Module = nn.Softmax(dim=-1)
              ):
    super().__init__()
    self.attn_matrix = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
    self.similarity = similarity

  def forward(self, x: Tensor) -> Tensor:
    scalar = torch.sqrt(torch.tensor(x.shape[-1], device=x.device))
    prod = x @ self.attn_matrix @ x.transpose(-1, -2) / scalar
    scores = self.similarity(prod)
    out = scores @ x 
    return out