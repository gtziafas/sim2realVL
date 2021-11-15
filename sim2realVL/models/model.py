from ..types import *

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from opt_einsum import contract


class RNNContext(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.LSTM(inp_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        _, (hn, _) = self.rnn(x)
        return hn[-1, ...]


class MultiLabelRNNVG(nn.Module):
    def __init__(self, 
                 visual_encoder: nn.Module,
                 text_encoder: RNNContext,
                 fusion_dim: int,
                 num_fusion_layers: int,
                 visual_feat_dim: int = 526,
                 text_feat_dim: int = 300,
                 with_downsample: bool = True
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.d_f = fusion_dim
        self.num_f_layers = num_fusion_layers
        self.venc = visual_encoder
        self.tenc = text_encoder
        self.d_inp = vfeat_dim + tfeat_dim
        if with_downsample:
            self.downsample = nn.Linear(self.d_v + self.d_t, self.d_f)
            self.d_inp = self.d_f
        self.rnn = nn.LSTM(self.d_inp, self.d_f, self.num_f_layers, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(2 * self.d_f, 1)

    def forward(self, vseq: Tensor, tseq: Tensor) -> Tensor:
        # vseq: B x N x 1 x H x W, tseq: B x W x Dt
        batch_size, num_boxes, _, height, width = vseq.shape
        
        vfeats = self.venc(vseq.view(batch_size * num_boxes, 1, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        tcontext = self.tenc(tseq)  # B x Dt
        
        fusion = torch.cat((vfeats, tfeats), dim=-1) # B x N x (Dv+Dt)
        if self.with_downsample:
            fusion = self.downsample(fusion).tanh()  # B x N x Df
        fusion, _ = self.rnn(fusion)    # B x N x (2*Df)
        out = self.cls(fusion)  # B x N x 1

    @torch.no_grad()
    def predict_scores(self, vseq: List[array], tseq: List[str]) -> Tensor:
        pass

    @torch.no_grad()
    def predict(self, vseq: List[array], tseq: List[str]) -> int:
        pass


class MultiLabelMHAVG(nn.Module):
    def __init__(self, 
                 visual_encoder: nn.Module,
                 fusion_kwargs: Dict[str, Any],
                 num_heads: int,
                 visual_feat_dim: int = 526,
                 text_feat_dim: int = 300,
                 with_downsample: bool = True
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.d_f = fusion_kwargs['fusion_dim']
        assert self.d_f % num_heads == 0, 'Must use #heads that divide fusion dim'
        self.d_a = self.d_f // num_heads
        self.num_heads = num_heads
        self.venc = visual_encoder
        self.w_q = nn.Linear(self.d_t, self.d_f)
        self.w_k = nn.Linear(self.d_v, self.d_f)
        self.w_v = nn.Linear(self.d_v, self.d_f)
        self.cls = nn.Linear(self.d_f, 1)

    def multi_head_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: MayTensor = None) -> Tensor:
        # q: B x N x Da x H,    k,v: B x W x Da x H,    Df = H * Da
        batch_size, num_boxes, model_dim, num_heads = q.shape
        dividend = torch.sqrt(torch.tensor(model_dim, device=q.device, dtype=floatt))

        weights = contract('bndh,bldh->bnlh', q, k) # B x N x W x H 
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
            weights = weights.masked_fill_(mask == 0, value=-1e-10)
        probs = weights.softmax(dim=-2) # B x N x W x H

        return contract('bnlh,bldh->bndh', probs, v).flatten(-2) # B x N x Df

    def forward(self, vseq: Tensor, tseq: Tensor) -> Tensor:
        # vseq: B x N x RGB, tseq: B x W x Dt
        batch_size, num_boxes, _, height, width = vseq.shape
        num_words = tseq.shape[1]
        
        vfeats = self.venc(vseq.view(batch_size * num_boxes, 3, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1)
        
        v_proj = self.w_q(vfeats).view(batch_size, num_boxes, -1, self.num_heads)
        t_keys = self.w_k(tseq).view(batch_size, num_words, -1, self.num_heads)
        t_vals = self.w_v(tseq).view(batch_size, num_words, -1, self.num_heads)
        attended = self.multi_head_attention(v_proj, t_keys, t_vals)

        return self.cls(attended)

    @torch.no_grad()
    def predict_scores(self, vseq: List[array], tseq: List[str]) -> Tensor:
        pass

    @torch.no_grad()
    def predict(self, vseq: List[array], tseq: List[str]) -> int:
        pass

class ClosedVQA(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, ):
        pass


class OpenVQA(nn.Module):
    def __init__(self, ):
        pass

    def forward(self, ):
        pass


class VE(nn.Module):
    def __init__(self, ):
        pass

    def forward(self, ):
        pass


def collate_images(device: str, 
                   with_padding: Maybe[Tuple[int, int]] = None,
                   with_labels: bool = False
                   ) -> Map[Sequence[Object], Tuple[Tensor, MayTensor]]:
    
    def _collate(batch: Sequence[Object]) -> Tuple[Tensor, MayTensor]:
        imgs, labels = zip(*[(o.image, o.label) for o in batch])
    
        # pad to equal size if desired
        if with_padding is not None:
            imgs = pad_with_frame(imgs, desired_shape=with_padding)

        # normalize to [0,1], tensorize, send to device, add channel dimension and stack
        imgs = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs]).unsqueeze(1)

        # tensorize labels if desired
        labels = stack([tensor(label, dtype=longt, device=device) for label in labels]) if with_labels else None
        
        return imgs, labels

    return _collate


def collate_text(device: str, pad_token_id: int = 0) -> Map[Sequence[array], Tensor]:
    
    def _collate(batch: Sequence[array]) -> Tensor:
        return pad_sequence(batch, batch_size=True, padding_value=pad_token_id)

    return _collate
