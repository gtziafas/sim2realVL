from ..types import *
from ..models.visual_embedder import make_visual_embedder, custom_features
from ..models.position_embedder import make_position_embedder
from ..models.nets import GRUContext
from ..models.fusion import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from opt_einsum import contract


# template
class MultiLabelVG(nn.Module):
    def __init__(self, 
                 visual_embedder: Maybe[nn.Module],
                 text_embedder: nn.Module,
                 position_embedder: nn.Module,
                 fusion: nn.Module,
                 classifier: nn.Module,
                 dropout: float = .0):
        super().__init__()
        self.vis_emb = visual_embedder
        self.text_emb = text_embedder
        self.pos_emb = position_embedder
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout)
        self.classifier = classifier

        # if skipping visual encoder, run twostage forward method
        self.forward = self._forward_twostage if visual_embedder is None else self._forward_onestage

    def _forward_onestage(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        visual, text, position = inputs
        # visual: B x N x 3 x H x W, position: B x N x Dp, text: B x T x Dt
        assert self.vis_emb is not None
        batch_size, num_objects, _, height, width = visual.shape
        
        vis_feats = self.vis_emb(visual.view(batch_size * num_objects, 3, height, width))
        vis_feats = vis_feats.view(batch_size, num_objects, -1) # B x N x Dv

        text_context = self.text_emb(text).unsqueeze(1).repeat(1, num_objects, 1) # B x N xDt
        
        position = self.pos_emb(position)

        fused = self.fusion((vis_feats, position, text_context))  # B x N x Df
        fused = self.dropout(fused)
        
        return self.classifier(fused).squeeze() # B x N x 1

    def _forward_twostage(self, inputs: Tuple[Tensor, ...]):
        vis_feats, text, position = inputs
        # vis_feats: B x N x Dv, position: B x N x Dp, text: B x T x Dt
        assert self.vis_emb is None
        batch_size, num_objects = vis_feats.shape[0:2]
        
        text_context = self.text_emb(text).unsqueeze(1).repeat(1, num_objects, 1) # B x N xDt
        
        position = self.pos_emb(position)

        fused = self.fusion((vis_feats, position, text_context))  # B x N x Df
        fused = self.dropout(fused)
        
        return self.classifier(fused).squeeze() # B x N x 1      


# # M2: Cross-attention between bounding boxes and words
# class MultiLabelAttentionVG(nn.Module):
#     def __init__(self, 
#                  visual_embedder: Maybe[nn.Module],
#                  position_encoder: nn.Module,
#                  text_encoder: RNNContext,
#                  hidden_dim: int,
#                  fusion_dim: int,
#                  num_heads: int = 1,  
#                  visual_feat_dim: int = 256,
#                  text_feat_dim: int = 300,
#                  position_feat_dim: int = 4,
#                  dropout: float = 0.33
#                  ):
#         super().__init__()
#         self.d_v = visual_feat_dim
#         self.d_t = text_feat_dim
#         self.d_f = fusion_dim
#         assert self.d_f % num_heads == 0, 'Must use #heads that divide fusion dim'
#         self.d_a = self.d_f // num_heads
#         self.d_p = position_feat_dim
#         self.num_heads = num_heads
#         self.venc = visual_embedder
#         self.penc = position_encoder
#         self.tenc = text_encoder
#         self.w_q = nn.Linear(self.d_v + self.d_p + self.d_t, self.d_f)
#         self.w_k = nn.Linear(self.d_v + self.d_p + self.d_t, self.d_f)
#         self.w_v = nn.Linear(self.d_v + self.d_p + self.d_t, self.d_f)
#         self.w_o = nn.Linear(self.d_f, self.d_f)
#         #self.cls = nn.Linear(self.d_f, 1)
#         self.cls = nn.Sequential(nn.Linear(self.d_f, hidden_dim),
#                                  nn.GELU(),
#                                  nn.Linear(hidden_dim, 1))
#         self.dropout = nn.Dropout(dropout)

#         # if skipping visual encoder, run other forward method
#         self.forward = self._forward_fast if visual_embedder is None else self._forward

#     def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: MayTensor = None) -> Tensor:
#         # q: B x N x Da x H,    k,v: B x T x Da x H,    Df = H * Da
#         batch_size, num_boxes, model_dim, num_heads = q.shape
#         dividend = torch.sqrt(torch.tensor(model_dim * num_heads, device=q.device, dtype=floatt))

#         weights = contract('bndh,btdh->bnth', q, k).div(dividend) # B x N x T x H 
#         if mask is not None:
#             mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
#             weights = weights.masked_fill_(mask == 0, value=-1e-10)
#         probs = weights.softmax(dim=-2) # B x N x T x H
#         return contract('bnth,btdh->bndh', probs, v).flatten(-2) # B x N x Df


#     def _forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
#         visual, text, position = inputs
#         batch_size, num_boxes, _, height, width = visual.shape

#         # visual: B x N x RGB, text: B x T x Dt
#         tcontext = self.tenc(text).unsqueeze(1).repeat(1, vfeats.shape[1], 1)  # B x N x Dt
        
#         vfeats = self.venc(visual.view(batch_size * num_boxes, 3, height, width))
#         vfeats = vfeats.view(batch_size, num_boxes, -1) # B x N x Dv
        
#         position = self.penc(position)
        
#         catted = torch.cat((vfeats, position, tcontext), dim=-1) 
#         attended = self.mha(catted, catted, catted)
#         attended = self.dropout(attended)
#         return self.cls(attended).squeeze()

#     def _forward_fast(self, inputs: Tuple[Tensor, ...]) -> Tensor:
#         vfeats, text, position = inputs
#         position = self.penc(position)
#         tcontext = self.tenc(text).unsqueeze(1).repeat(1, vfeats.shape[1], 1)  # B x N x Dt
#         # vfeats: B x N x Dv, text: B x T x Dt
#         catted = torch.cat((vfeats, position, tcontext), dim=-1) 
#         attended = self.mha(catted, catted, catted)
#         attended = self.dropout(attended)
#         return self.cls(attended).squeeze()

#     def mha(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
#         batch_size, num_objects = queries.shape[0:2]

#         q_proj = self.w_q(queries).view(batch_size, num_objects, -1, self.num_heads)
#         k_proj = self.w_k(keys).view(batch_size, num_objects, -1, self.num_heads)
#         v_proj = self.w_v(values).view(batch_size, num_objects, -1, self.num_heads)
#         scores =  self.attention(q_proj, k_proj, v_proj)
#         return self.w_o(scores)


def collate(device: str, ignore_idx: int = -1) -> Map[List[Tuple[Tensor, ...]], Tuple[Tensor, ...]]:
    def _collate(batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        visual, text, truths, position = zip(*batch)
        visual = pad_sequence(visual, batch_first=True, padding_value=ignore_idx).to(device)
        text = pad_sequence(text, batch_first=True).to(device)
        truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx).to(device)
        position = pad_sequence(position, batch_first=True, padding_value=ignore_idx).to(device) 
        return visual, text, truths, position
    return _collate


def make_model(model_id: str, position_emb: str = "raw", stage: int = 2):
    pe = make_position_embedder(position_emb)
    position_feats_dim = pe.num_features 
    
    ve = custom_features() if stage == 1 else None 

    if model_id == "MLP":
        te = GRUContext(300, 300, 1)   
        fusion_dim =  256 + 300 + position_feats_dim
        fusion = ConcatFusion(fusion_dim=fusion_dim)
        head = nn.Sequential(nn.Linear(fusion_dim, 128),
                             nn.GELU(),
                             nn.Linear(128, 1))

    elif model_id == "RNN":
        te = GRUContext(300, 300, 1)
        fusion_dim = 256
        fusion = GRUFusion(input_dim=256 + 300 + position_feats_dim, 
                fusion_dim=fusion_dim, 
                hidden_dim=fusion_dim)

        head = nn.Linear(fusion_dim, 1)
        # head = nn.Sequential(nn.Linear(fusion_dim, 128),
        #                      nn.GELU(),
        #                      nn.Linear(128, 1))

    elif model_id == "CNN":
        te = GRUContext(300, 256, 1)
        fusion = Conv1x1Fusion(fusion_dim=256)
        head = nn.Linear(256, 1)

    else:
        raise ValueError("Check models.vg for options")

    return MultiLabelVG(visual_embedder=ve,
                        text_embedder=te,
                        position_embedder=pe,
                        fusion=fusion,
                        classifier=head
                        )