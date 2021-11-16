from ..types import *
from ..utils.word_embedder import WordEmbedder, make_word_embedder
from ..utils.image_proc import crop_boxes_fixed
from ..data.rgbd_scenes import RGBDScenesVG
from ..models.visual_embedder import make_visual_embedder, custom_features

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from opt_einsum import contract
from PIL import Image


class RNNContext(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.LSTM(inp_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        dh = self.rnn.hidden_size
        H, _ = self.rnn(x)  # B x T x 2*D
        return torch.cat((H[:, -1, :dh], H[:, 0, dh:]), dim=-1) # B x 2*D


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
                 visual_feat_dim: int = 256,
                 hidden_dim: int = 128,
                 text_feat_dim: int = 300,
                 dropout: float = 0.33
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.tenc = text_encoder
        self.venc = visual_encoder
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(self.d_v + self.d_t + 4, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, 1))
        #self.cls = nn.Linear(self.d_v + self.d_t + 4, 1)

        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        visual, text, boxes = inputs
        # visual: B x N x 3 x H x W, boxes: B x N x 4, text: B x T x Dt
        assert self.venc is not None
        batch_size, num_boxes, _, height, width = visual.shape
        
        vfeats = self.venc(visual.view(batch_size * num_boxes, 3, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        tcontext = self.tenc(text).unsqueeze(1).repeat(1, num_boxes, 1) # B x N xDt

        catted = torch.cat((vfeats, boxes, tcontext), dim=-1) # B x N x (Dv+4+Dt)
        catted = self.dropout(catted)
        return self.cls(catted).squeeze() # B x N x 1

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        # vfeats: B x N x Dv, boxes: B x N x 4, text: B x T x Dt
        visual, text, boxes = inputs
        #boxes[:, :, :] = 0. # -> witness if 2d info helps

        #print(visual.shape, text.shape, boxes.shape)
        tcontext = self.tenc(text).unsqueeze(1)
        #print(tcontext.shape)
        tcontext = tcontext.repeat(1, visual.shape[1], 1)
        #print(tcontext.shape)
        catted = torch.cat((visual, boxes, tcontext), dim=-1)
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
        self.d_inp = self.d_v + self.d_t + 4
        self.with_downsample = with_downsample
        if with_downsample:
            self.downsample = nn.Linear(self.d_inp, self.d_f)
            self.d_inp = self.d_f
        self.rnn = nn.LSTM(self.d_inp, self.d_f, self.num_f_layers, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(2 * self.d_f, 1)

        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        visual, text, boxes = inputs

        # visual: B x N x 1 x H x W, text: B x T x Dt
        assert self.venc is not None
        batch_size, num_boxes, _, height, width = visual.shape
        
        vfeats = self.venc(visual.view(batch_size * num_boxes, 3, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        tcontext = self.tenc(text).unsqueeze(1).repeat(1, vfeats.shape[1], 1)
        return self.fuse(vfeats, tcontext, boxes)

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vfeats, text, boxes = inputs
        # vfeats: B x N x Dv, text: B x T x Dt
        tcontext = self.tenc(text).unsqueeze(1).repeat(1, vfeats.shape[1], 1)  # B x N x Dt
        return self.fuse(vfeats, tcontext, boxes)

    def fuse(self, vfeats: Tensor, tcontext: Tensor, boxes: Tensor) -> Tensor:
        #boxes[...] = 0.1
        fusion = torch.cat((vfeats, tcontext, boxes), dim=-1) # B x N x (Dv+Dt+4)
        if self.with_downsample:
            fusion = self.downsample(fusion).tanh()  # B x N x Df
        fusion, _ = self.rnn(fusion)    # B x N x (2*Df)
        return self.cls(fusion).squeeze()  # B x N


# M2: Cross-attention between bounding boxes and words
class MultiLabelMHAVG(nn.Module):
    def __init__(self, 
                 visual_encoder: Maybe[nn.Module],
                 fusion_dim: int,
                 num_heads: int,  
                 visual_feat_dim: int = 512,
                 text_feat_dim: int = 300,
                 ):
        super().__init__()
        self.d_v = visual_feat_dim
        self.d_t = text_feat_dim
        self.d_f = fusion_dim
        assert self.d_f % num_heads == 0, 'Must use #heads that divide fusion dim'
        self.d_a = self.d_f // num_heads
        self.num_heads = num_heads
        self.venc = visual_encoder
        self.tenc = nn.LSTM(self.d_t, self.d_f // 2, 1, bidirectional=True, batch_first=True)
        self.w_q = nn.Linear(self.d_v + 4, self.d_f)
        self.w_k = nn.Linear(self.d_t, self.d_f)
        self.w_v = nn.Linear(self.d_t, self.d_f)
        self.cls = nn.Linear(self.d_f, 1)

        # if skipping visual encoder, run other forward method
        self.forward = self._forward_fast if visual_encoder is None else self._forward

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: MayTensor = None) -> Tensor:
        # q: B x N x Da x H,    k,v: B x T x Da x H,    Df = H * Da
        batch_size, num_boxes, model_dim, num_heads = q.shape
        dividend = torch.sqrt(torch.tensor(model_dim * num_heads, device=q.device, dtype=floatt))

        weights = contract('bndh,btdh->bnth', q, k) # B x N x T x H 
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
            weights = weights.masked_fill_(mask == 0, value=-1e-10)
        probs = weights.softmax(dim=-2) # B x N x T x H

        return contract('bnlh,btdh->bndh', probs, v).flatten(-2) # B x N x Df

    def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        visual, text, boxes = inputs
        # visual: B x N x RGB, text: B x T x Dt
        tfeats, _ = self.tenc(text)
        batch_size, num_boxes, _, height, width = visual.shape
        vfeats = self.venc(visual.view(batch_size * num_boxes, 3, height, width))
        vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        return self.mha(torch.cat((vfeats, boxes), dim=-1), tfeats)

    def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        vfeats, text, boxes = inputs
        tfeats, _ = self.tenc(text)
        # vfeats: B x N x Dv, text: B x T x Dt
        return self.mha(torch.cat((vfeats, boxes), dim=-1), tfeats)

    def mha(self, vfeats: Tensor, tfeats: Tensor) -> Tensor:
        batch_size, num_words = tfeats.shape[0:2]
        num_boxes = vfeats.shape[1]

        v_proj = self.w_q(vfeats).view(batch_size, num_boxes, -1, self.num_heads)
        t_keys = self.w_k(tfeats).view(batch_size, num_words, -1, self.num_heads)
        t_vals = self.w_v(tfeats).view(batch_size, num_words, -1, self.num_heads)
        attended = self.attention(v_proj, t_keys, t_vals)
        return self.cls(attended).squeeze()


def collate(device: str, ignore_idx: int = -1) -> Map[List[Tuple[Tensor, ...]], Tuple[Tensor, Tensor, Tensor, MayTensor]]:
    def _collate(batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, Tensor, Tensor, MayTensor]:
        visual, text, truths, position = zip(*batch)
        visual = pad_sequence(visual, batch_first=True, padding_value=ignore_idx).to(device)
        text = pad_sequence(text, batch_first=True).to(device)
        truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx).to(device)
        position = pad_sequence(position, batch_first=True, padding_value=ignore_idx).to(device) 
        return visual, text, truths, position
    return _collate


def onestage_vg_model_rnn():
  from torchvision.models import resnet18
  venc = resnet18()
  venc.fc = torch.nn.Identity()
  return MultiLabelRNNVG(visual_encoder=venc, text_encoder=RNNContext(300, 150, 1), 
                                fusion_dim=200, num_fusion_layers=1, with_downsample=True)

def onestage_vg_model(ve: nn.Module = custom_features()):
  return MultiLabelMLPVG(visual_encoder=ve, text_encoder=RNNContext(300, 150, 1))  


def twostage_vg_model_rnn():
  return MultiLabelRNNVG(visual_encoder=None, text_encoder=RNNContext(300, 150, 1), 
                                fusion_dim=200, num_fusion_layers=1, with_downsample=True)

def twostage_vg_model():
  return MultiLabelMLPVG(visual_encoder=None, text_encoder=RNNContext(300, 150, 1))



def make_vg_dataset(ds: RGBDScenesVG, 
                  fast : Maybe[str] = None,
                  save_path: Maybe[str] = None,   
                  img_resize: Tuple[int, int] = (120, 120),
                  scene_shape: Tuple[int, int] = (640, 480)
                  ) -> List[Tuple[Tensor, ...]]:
  W, H = scene_shape

  from ..utils.word_embedder import make_word_embedder
  we = make_word_embedder()
  #querries = we([ds[i].query for i in range(len(ds))])

  # split indices into groups of unique scenes
  idces = [[i for i, p in enumerate(ds.rgb_paths) if p == path] for path in ds.unique_scenes]

  tensorized = []
  print('Tensorizing data...')
  if not fast:
    for ids in idces:
       crops = ds[ids[0]].get_crops()
       crops = crop_boxes_fixed(img_resize)(crops)
       crops = torch.stack([torch.tensor(c, dtype=floatt) for c in crops]).view(-1, 3, *img_resize)

       for index in ids:
          scene = ds[index]
          query = we([scene.query])[0]
          query = torch.tensor(query, dtype=floatt)
          truth = torch.tensor(scene.truth, dtype=longt)
          boxes = torch.stack([torch.tensor([b.x/W, b.y/H, b.w/W, b.h/H]).float() for b in scene.boxes])

          tensorized.append((crops, query, truth, boxes))
  else:
    all_feats = torch.load(fast)
    for i, scene in enumerate(ds):
      feats = all_feats[i]
      query = we([scene.query])[0]
      query = torch.tensor(query, dtype=floatt)
      truth = torch.tensor(scene.truth, dtype=longt)
      boxes = torch.stack([torch.tensor([b.x/W, b.y/H, b.w/W, b.h/H]).float() for b in scene.boxes])

      tensorized.append((feats, query, truth, boxes))      

  if save_path is not None:
    print(f'Saving checkpoint in {save_path}')
    torch.save(tensorized, save_path)
  return tensorized