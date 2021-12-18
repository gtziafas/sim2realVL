from ..types import *
from ..models.visual_embedder import make_visual_embedder, custom_features
from ..models.position_embedder import make_position_embedder
from ..models.nets import *
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
                 dropout: float):
        super().__init__()
        self.vis_emb = visual_embedder
        self.text_emb = text_embedder
        self.pos_emb = position_embedder
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout)
        self.classifier = classifier

        # if skipping visual encoder, run twostage forward method
        self.forward = self._forward_twostage if visual_embedder is None else self._forward_onestage

    def _forward_onestage(self, visual: Tensor, queries: Tensor, position: Tensor) -> Tensor:
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

    def _forward_twostage(self, visual: Tensor, queries: Tensor, position: Tensor) -> Tensor:
        vis_feats = visual
        assert self.vis_emb is None
        batch_size, num_objects = vis_feats.shape[0:2]
        
        text_context = self.text_emb(text).unsqueeze(1).repeat(1, num_objects, 1) # B x N xDt
        
        position = self.pos_emb(position)

        fused = self.fusion((vis_feats, position, text_context))  # B x N x Df
        fused = self.dropout(fused)
        
        return self.classifier(fused).squeeze() # B x N x 1      


def collate(device: str, without_position: bool = False, ignore_idx: int = -1) -> Map[List[Tuple[Tensor, ...]], Tuple[Tensor, ...]]:
    def _collate(batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        visual, text, truths, position = zip(*batch)
        visual = pad_sequence(visual, batch_first=True, padding_value=ignore_idx).to(device)
        text = pad_sequence(text, batch_first=True).to(device)
        truths = pad_sequence(truths, batch_first=True, padding_value=ignore_idx).to(device)
        position = pad_sequence(position, batch_first=True, padding_value=ignore_idx).to(device) 
        return (visual, text, position, truths) if not without_position else (visual, text, truths)
    return _collate


def make_model(model_id: str, position_emb: str = "raw", stage: int = 2):
    pe = make_position_embedder(position_emb, with_attention=None)
    position_feats_dim = pe.num_features 
    
    ve = custom_features() if stage == 1 else None 

    if model_id == "Test":
        te = GRUContext(300, 300, 1)
        fusion_dim = 512
        fusion = RelativeExtensionFusion(fusion_dim=fusion_dim,
                text_feat_dim=300,
                visual_feat_dim=256,
                position_feat_dim=position_feats_dim,
                dropout=0.33)
        head = nn.Identity()
        dropout = 0.

    elif model_id == "MLP":
        te = GRUContext(300, 300, 1)   
        fusion_dim =  256 + 300 + position_feats_dim
        # fusion = nn.Sequential(ConcatFusion(fusion_dim),
        #                        nn.Linear(fusion_dim, 512),
        #                        nn.GELU(),
        #                      )
        # head = nn.Linear(512, 1)
        # dropout = 0.33
        fusion = ConcatFusion(fusion_dim)
        head = nn.Sequential(nn.Linear(fusion_dim, 512),
                             nn.GELU(),
                             nn.Dropout(0.33),
                             nn.Linear(512, 1))
        dropout = 0.

    elif model_id == "Attention":
        te = GRUContext(300, 300, 1)   
        fusion_dim = 512 
        fusion = CrossModalAttentionFusion(fusion_dim=fusion_dim,
                text_feat_dim=300,
                visual_feat_dim=256,
                position_feat_dim=position_feats_dim)
        #head = nn.Linear(512, 1)
        head = nn.Sequential(nn.Linear( 256 + 300 + position_feats_dim, 512),
                             nn.GELU(),
                             nn.Dropout(0.33),
                             nn.Linear(512, 1))
        dropout = 0.0

    elif model_id == "RNN":
        te = GRUContext(300, 300, 1)
        fusion_dim = 512
        fusion = GRUFusion(input_dim=256 + 300 + position_feats_dim, 
                fusion_dim=fusion_dim, 
                hidden_dim=fusion_dim,
                bidirectional=True)

        #head = nn.Linear(fusion_dim, 1)
        dropout = 0.33
        head = nn.Sequential(nn.Linear(fusion_dim, 256),
                             nn.GELU(),
                             nn.Dropout(0.5),
                             nn.Linear(256, 1))

    elif model_id == "CNN":
        te = GRUContext(300, 300, 1)
        fusion = Conv1x1Fusion(fusion_dim=256)
        head = nn.Linear(256, 1)
        dropout =0.

    else:
        raise ValueError("Check models.vg for options")

    return MultiLabelVG(visual_embedder=ve,
                        text_embedder=te,
                        position_embedder=pe,
                        fusion=fusion,
                        classifier=head,
                        dropout=dropout
                        )