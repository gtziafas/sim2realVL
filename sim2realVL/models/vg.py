from ..types import *
from ..utils.word_embedder import WordEmbedder

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from opt_einsum import contract


class RNNContext(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.LSTM(inp_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        H, _ = self.rnn(x)  # B x T x 2*D
        return torch.cat((H[:, -1, :self.rnn.hidden_size], H[:, 0, self.rnn.hidden_size:]), dim=-1) # B x 2*D


# template
class MultiLabelVG(nn.Module):

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        ...

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        ...

    @torch.no_grad()
    def predict_scores(self, scene: Scene, query: str) -> Tensor:
        ...

    @torch.no_grad()
    def predict(self, scene: Scene, query: str) -> int:
        ...

    def load_pretrained(path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)


# M0: RNNContext for language + MLP for bounding boxes + 2dPosVector + context
class MultiLabelMLPVG(MultiLabelVG):
    def __init__(self, 
                 visual_encoder: Maybe[nn.Module],
                 text_encoder: RNNContext,
                 visual_feat_dim: int = 512,
                 text_feat_dim: int = 300,
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.tenc = text_encoder
        self.venc = visual_encoder
        self.cls = nn.Sequential(nn.Linear(self.d_v + self.d_t + 4, 256),
                                 nn.GELU(),
                                 nn.Linear(256, 1))


        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vseq, tseq, bseq = inputs
        # vseq: B x N x 1 x H x W, bseq: B x N x 4, tseq: B x T x Dt
        assert self.venc is not None
        batch_size, num_boxes, _, height, width = vseq.shape
        
        vfeats = self.venc(vseq.view(batch_size * num_boxes, 1, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        tcontext = self.tenc(tseq).unsqueeze(1).repeat(1, num_boxes, 1) # B x N xDt

        catted = torch.cat((vfeats, bseq, tcontext), dim=-1) # B x N x (Dv+4+Dt)
        return self.cls(catted).squeeze() # B x N x 1

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        # vfeats: B x N x Dv, bseq: B x N x 4, tseq: B x T x Dt
        vfeats, tseq, bseq = inputs
        bseq[:, :, :] = 0.5 # -> witness if 2d info helps

        #print(vfeats.shape, tseq.shape, bseq.shape)
        tcontext = self.tenc(tseq).unsqueeze(1)
        #print(tcontext.shape)
        tcontext = tcontext.repeat(1, vfeats.shape[1], 1)
        #print(tcontext.shape)
        catted = torch.cat((vfeats, bseq, tcontext), dim=-1)
        #print(catted.shape)
        return self.cls(catted).squeeze()


# M1: RNNContext for language + RNN for bounding boxes + context
class MultiLabelRNNVG(MultiLabelVG):
    def __init__(self, 
                 visual_encoder: Maybe[nn.Module],
                 text_encoder: RNNContext,
                 fusion_dim: int,
                 num_fusion_layers: int,
                 visual_feat_dim: int = 512,
                 text_feat_dim: int = 300,
                 with_downsample: bool = True,
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.d_f = fusion_dim
        self.num_f_layers = num_fusion_layers
        self.tenc = text_encoder
        self.venc = visual_encoder
        self.d_inp = vfeat_dim + tfeat_dim
        if with_downsample:
            self.downsample = nn.Linear(self.d_v + self.d_t, self.d_f)
            self.d_inp = self.d_f
        self.rnn = nn.LSTM(self.d_inp, self.d_f, self.num_f_layers, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(2 * self.d_f, 1)

        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vseq, tseq, _ = inputs

        # vseq: B x N x 1 x H x W, tseq: B x T x Dt
        assert self.venc is not None
        batch_size, num_boxes, _, height, width = vseq.shape
        
        vfeats = self.venc(vseq.view(batch_size * num_boxes, 1, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        tcontext = self.tenc(tseq)
        return self.fuse(vfeats, tcontext)

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vfeats, tseq = inputs
        # vfeats: B x N x Dv, tseq: B x T x Dt
        tcontext = self.tenc(tseq)  # B x Dt
        return self.fuse(vfeats, tcontext)

    def fuse(self, vfeats: Tensor, tcontext: Tensor) -> Tensor:
        fusion = torch.cat((vfeats, tcontext), dim=-1) # B x N x (Dv+Dt)
        if self.with_downsample:
            fusion = self.downsample(fusion).tanh()  # B x N x Df
        fusion, _ = self.rnn(fusion)    # B x N x (2*Df)
        return self.cls(fusion).squeeze()  # B x N


# M2: Cross-attention between bounding boxes and words
class MultiLabelMHAVG(nn.Module):
    def __init__(self, 
                 visual_encoder: nn.Module,
                 fusion_kwargs: Dict[str, Any],
                 num_heads: int,
                 visual_feat_dim: int = 512,
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

        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: MayTensor = None) -> Tensor:
        # q: B x N x Da x H,    k,v: B x W x Da x H,    Df = H * Da
        batch_size, num_boxes, model_dim, num_heads = q.shape
        dividend = torch.sqrt(torch.tensor(model_dim, device=q.device, dtype=floatt))

        weights = contract('bndh,bldh->bnlh', q, k) # B x N x W x H 
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
            weights = weights.masked_fill_(mask == 0, value=-1e-10)
        probs = weights.softmax(dim=-2) # B x N x W x H

        return contract('bnlh,bldh->bndh', probs, v).flatten(-2) # B x N x Df

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vseq, tseq, _ = inputs
        # vseq: B x N x RGB, tseq: B x W x Dt
        batch_size, num_boxes, _, height, width = vseq.shape
        vfeats = self.venc(vseq.view(batch_size * num_boxes, 3, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        return self.mha(vfeats, tseq)

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vfeats, tseq, _ = inputs
        # vfeats: B x N x Dv, tseq: B x W x Dt
        return self.mha(vfeats, tseq)

    def mha(self, vfeats: Tensor, tseq: Tensor) -> Tensor:
        batch_size, num_words = tfeats.shape[0:2]
        num_boxes = vfeats.shape[1]

        v_proj = self.w_q(vfeats).view(batch_size, num_boxes, -1, self.num_heads)
        t_keys = self.w_k(tseq).view(batch_size, num_words, -1, self.num_heads)
        t_vals = self.w_v(tseq).view(batch_size, num_words, -1, self.num_heads)
        attended = self.attention(v_proj, t_keys, t_vals)

        return self.cls(attended)


def collate(device: str, 
            word_embedder: WordEmbedder,
            with_boxes: bool = True, 
            ignore_idx: int = -1,
            scene_shape: Tuple[int, int] = (640, 480),
            with_normalization: bool = True
            ) -> Map[Sequence[AnnotatedScene], Tuple[Tensor, Tensor, Tensor, MayTensor]]:
    W, H = scene_shape
    tf = T.Compose([T.CenterCrop(224), T.ToTensor()])
    if with_normalization:
        tf = T.Compose([tf, T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _collate(batch: Sequence[AnnotatedScene]) -> Tuple[Tensor, Tensor, Tensor, MayTensor]:
        imgs = [scene.get_crops() for scene in batch]
        imgs = [[Image.fromarray(c) for c in crops] for crops in imgs]
        imgs = [torch.stack([tf(c) for c in crops]) for crops in imgs]
        imgs = pad_sequence(imgs, batch_first=True, padding_value=ignore_idx).to(device)

        words, truths, boxes = zip(*[(s.query, s.truth, s.boxes) for s in batch])
        words = word_embedder(words)
        words = [torch.tensor(ws, dtype=floatt, device=device) for ws in words]
        words = pad_sequence(words, batch_first=True)

        truths = [torch.tensor(t, dtype=longt, device=device) for t in truths]
        truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx)

        _boxes = None
        if with_boxes:
          _boxes = [torch.stack([torch.tensor([b.x/W, b.y/H, b.w/W, b.h/H], dtype=floatt, device=device) for b in bs]) for bs in boxes]
          _boxes = pad_sequence(_boxes, batch_first=True, padding_value=ignore_idx)
        return imgs, words, truths, _boxes

    return _collate


def collate_fast(device: str, 
            with_boxes: bool = True, 
            ignore_idx: int = -1,
            scene_shape: Tuple[int, int] = (640, 480),
            with_normalization: bool = True
            ) -> Map[Sequence[Tuple[Tensor, List[Tensor], List[Tuple[int]], List[Box]]], Tuple[Tensor, Tensor, Tensor, MayTensor]]:
    W, H = scene_shape

    def _collate(batch: Sequence[Tuple[Tensor, List[Tensor], List[Tuple[int]], List[Box]]]) -> Tuple[Tensor, Tensor, Tensor, MayTensor]:
        feats, words, truths, boxes = zip(*[(f, q, t, b) for f, q, t, b in batch])
        feats = pad_sequence(feats, batch_first=True, padding_value=ignore_idx).to(device)

        #words = word_embedder(words)
        words = pad_sequence([torch.tensor(w, dtype=floatt, device=device) for w in words], batch_first=True)

        truths = [torch.tensor(t, dtype=longt, device=device) for t in truths]
        truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx)

        _boxes = None
        if with_boxes:
          _boxes = [torch.stack([torch.tensor([b.x/W, b.y/H, b.w/W, b.h/H], dtype=floatt, device=device) for b in bs]) for bs in boxes]
          _boxes = pad_sequence(_boxes, batch_first=True, padding_value=ignore_idx)
        return feats, words, truths, _boxes

    return _collate


def default_vg_model():
  return MultiLabelMLPVG(visual_encoder=None, text_encoder=RNNContext(300, 150, 1))