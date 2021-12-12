from ..types import *
import torch
import torch.nn as nn 
import torch.nn.functional as F

from ..models.position_embedder import make_position_embedder
from ..utils.loss import BCEWithLogitsIgnore


class NormalizeScale(nn.Module):
    def __init__(self, dim: int, init_norm=20):
      super(NormalizeScale, self).__init__()
      self.init_norm = init_norm
      self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom: Tensor) -> Tensor:
      # input is variable (n, dim)
      assert isinstance(bottom, torch.autograd.Variable), 'bottom must be variable'
      bottom_normalized = nn.functional.normalize(bottom, p=2, dim=-1)
      bottom_normalized_scaled = bottom_normalized * self.weight
      return bottom_normalized_scaled


class Matching(nn.Module):
    def __init__(self, 
                 object_dim: int, 
                 text_dim: int, 
                 jemb_dim: int, 
                 jemb_dropout: float=0., 
                 with_bn: bool=False
                ):
        super().__init__()
        self.bn = nn.Identity() if not with_bn else nn.BatchNorm1d(jemb_dim)
        self.object_fc = nn.Sequential(nn.Linear(object_dim, jemb_dim),
                                    self.bn,
                                    nn.ReLU(),
                                    nn.Dropout(jemb_dropout),
                                    nn.Linear(jemb_dim, jemb_dim),
                                    self.bn
                                  ).apply(self.init_weights)

        self.text_fc = nn.Sequential(nn.Linear(text_dim, jemb_dim),
                                    self.bn,
                                    nn.ReLU(),
                                    nn.Dropout(jemb_dropout),
                                    nn.Linear(jemb_dim, jemb_dim),
                                    self.bn
                                  ).apply(self.init_weights)

    def init_weights(self, net: nn.Module):
        if isinstance(net, nn.Linear):
            nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(0.01)

    def forward(self, object_input: Tensor, text_input: Tensor) -> Tensor:
        # objects: B x N x Do, text: B x Dt
        B, N = object_input.shape[0:2]

        object_feat = object_input.view(B * N, -1)

        object_emb = self.object_fc(object_feat)
        text_emb = self.text_fc(text_input)

        # l2-normalize
        object_emb_norm = F.normalize(object_emb, p=2, dim=1)
        text_emb_norm = F.normalize(text_emb, p=2, dim=1)

        # repeat text to match object emb tile
        object_emb_norm_tile = object_emb_norm.view(B, N, -1)
        text_emb_norm_tile = text_emb_norm.unsqueeze(1).repeat(1, N, 1)

        # compute matching scores
        scores = torch.sum(text_emb_norm_tile * object_emb_norm_tile, 2)

        return scores


class RelationModule(nn.Module):
    def __init(self, 
               pos_dim: int, 
               text_dim: int,
               jemb_dim: int,
               jemb_dropout: float = 0.,
               with_bn: bool = False
              ):
        super().__init__()
        self.match = Matching(2*pos_dim, text_dim, jemb_dim, jemb_dropout, with_bn)

    def forward(self, pos_embs: Tensor, q_spt: Tensor) -> Tensor:
        # pos_embs: B x N x Dp, q_spt: B x Dt
        batch_size, num_objects = pos_embs.shape[0:2]
        
        # N x N pair-wise embeddings, flattened
        p_tile = pos_embs.unsqueeze(2).repeat(1, 1, num_objects, 1)
        pos_embs = torch.cat([p_tile, p_tile.transpose(1,2)], dim=-1)  
        pos_embs = pos_embs.view(batch_size, num_objects**2, -1) # B x N^2 x 2*Dp

        scores = self.match(pos_embs, q_spt)
        return scores.view(batch_size, num_objects, num_objects)

        
class RNNEncoder(nn.Module):
  def __init__(self, 
               inp_dim: int, 
               hidden_dim: int, 
               num_layers: int, 
               bidirectional: bool = True,
               inp_dropout: float = 0.,
               rnn_dropout: float = 0.,
               rnn_type: str='gru'):
      super().__init__()
      self.bidirectional = bidirectional
      self.rnn = getattr(nn, rnn_type.upper())(inp_dim, hidden_dim, num_layers, dropout=rnn_dropout, 
        bidirectional=bidirectional, batch_first=True)
      self.dropout = nn.Dropout(p=inp_dropout)

  def forward(self, x: Tensor) -> Tensor:
      dh = self.rnn.hidden_size
      x = self.dropout(x)
      H, _ = self.rnn(x)  # B x T x (2*)D
      context = H[:, -1, :] if not self.bidirectional else torch.cat((H[:, -1, :dh], H[:, 0, dh:]), dim=-1)
      return H, context


class WordTagger(nn.Module):
  def __init__(self, 
               tag_vocab_size: int,
               emb_dim: int, 
               hidden_dim: int, 
               num_rnn_layers: int, 
               bidirectional: bool = True,
               inp_dropout: float = 0.,
               rnn_dropout: float = 0.,
               fc_dropout: float = 0.,
               rnn_type: str = 'gru',
               pretrain: bool = False,
               word_vocab_size: Maybe[int] = None,
               load_checkpoint: Maybe[str] = None
              ):
      super().__init__()
      self.tag_vocab_size = tag_vocab_size
      self.pretrain = pretrain
      if pretrain:
        self.embedding = nn.Identity()
      else:
        self.embedding = nn.Embedding(word_vocab_size, emb_dim, padding_idx=-1)

      self.encoder = RNNEncoder(emb_dim, hidden_dim, num_rnn_layers, bidirectional, inp_dropout, 
                                rnn_dropout, rnn_type)
      
      fc_input_dim = 2 * hidden_dim if bidirectional else hidden_dim
      
      self.fc = nn.Linear(fc_input_dim, tag_vocab_size)
      self.fc_dropout = nn.Dropout(fc_dropout)
      self.inp_dropout = nn.Dropout(inp_dropout)

      if load_checkpoint is not None:
        self.load_state_dict(load_checkpoint)

  def forward(self, queries: Tensor) -> Tensor:
      word_embs = self.embedding(queries)
      word_embs = self.inp_dropout(word_embs)
      
      word_states, phrase_context = self.encoder(word_embs)
      
      word_states = self.fc_dropout(word_states)
      word_tags = self.fc(word_states)
      
      return word_tags, phrase_context

  def encode_per_tag(self, queries: Tensor, pad_mask: Tensor) -> Tensor:
      # queries: B x T x Dt, pad_mask: B x T
      # tagset: {0: '<loc>', 1: '<obj>', 2: '<rel>', 3: '<subj>'}
      tags, _ = self.forward(queries)
      embs = []
      for tag_id in range(self.tag_vocab_size):
        is_tag = torch.logical_and(tags.argmax(-1) == tag_id, ~pad_mask)
        tag_mask = torch.where(is_tag.unsqueeze(2), queries, torch.zeros_like(queries))
        embs.append(tag_mask.mean(1))
      return embs


class CMN(nn.Module):
  def __init__(self, cfg: Dict[str, Any]):
    super().__init__()
    self.padding_value = cfg['padding_value']

    self.visual_norm = NormalizeScale(cfg['visual_embed_size'])
    self.visual_fc = nn.Linear(cfg['visual_embed_size'], cfg['jemb_size'])

    self.position_embedder = make_position_embedder()
    self.position_norm = NormalizeScale(cfg['position_embed_size'])
    self.position_fc = nn.Linear(cfg['position_embed_size'], cfg['jemb_size'])
    
    self.word_norm = NormalizeScale(cfg['word_embed_size'])
    self.word_fc = nn.Linear(cfg['word_embed_size'], cfg['jemb_size'])

    self.word_tagger = WordTagger(len(cfg['tagset']),
                                  cfg['word_embed_size'],
                                  cfg['word_hidden_size'],
                                  cfg['word_num_rnn_layers'],
                                  cfg['word_rnn_bidirectional'],
                                  cfg['word_input_dropout'],
                                  cfg['word_rnn_dropout'],
                                  cfg['word_fc_dropout'],
                                  cfg['word_rnn_type'],
                                  cfg['word_embed_pretrain'],
                                  cfg['word_vocab_size']
                                  )

    self.ground_mod = Matching(cfg['jemb_size'],
                               cfg['jemb_size'],
                               cfg['jemb_size'],
                               cfg['jemb_dropout'],
                               cfg['with_batch_norm']
                              )

    self.spatial_mod = Matching(2 * cfg['jemb_size'],
                                cfg['jemb_size'],
                                cfg['jemb_size'],
                                cfg['jemb_dropout'],
                                cfg['with_batch_norm']
                              )

    if cfg['word_tagger_pretrain'] is not None:
      self.word_tagger.load_state_dict(torch.load(cfg['word_tagger_pretrain']))

    self.strong = cfg['strong_supervision']

  def forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
    # visual: B x N x Dv, position: B x N x Dp, query: B x T x Dt
    visual, queries, position = inputs
    batch_size, num_objects = visual.shape[0:2]

    # normalize and project visual embs 
    visual = self.visual_fc(self.visual_norm(visual))

    # average word embeddigns according to predicted tag
    pad_mask = (queries == self.padding_value).sum(-1) == queries.shape[-1]
    qs = self.word_tagger.encode_per_tag(queries, pad_mask) # (4x) B x Dt
    q_loc, q_obj, q_rel, q_subj = self.word_fc(self.word_norm(torch.stack(qs, dim=0)))

    # unitary score subject
    subj_scores = self.ground_mod(visual, q_subj) # B x N 
    
    # update pair-wise scores by repeating subject row-wise
    S_subj = subj_scores.unsqueeze(2).repeat(1, 1, num_objects)
    
    # N x N pair-wise position embeddings, flattened
    position = self.position_embedder(position)
    position = self.position_fc(self.position_norm(position))
    p_tile = position.unsqueeze(2).repeat(1, 1, num_objects, 1)
    p_pair = torch.cat([p_tile, p_tile.transpose(1,2)], dim=-1)  
    p_pair = p_pair.view(batch_size, num_objects**2, -1) # B x N^2 x 2*Dp
    
    # in (absolute spatial) location target object is same category as subject
    loc_scores = self.spatial_mod(p_pair, q_loc).view(-1, num_objects, num_objects)
    loc_obj_scores = subj_scores

    # update pair-wise scores by repeating object column-wise and adding spatial scores
    S_loc = loc_obj_scores.unsqueeze(1).repeat(1, num_objects, 1) + loc_scores 

    # unitary object + pair-wise relation score (relative spatial) if given
    rel_scores = self.spatial_mod(p_pair, q_rel).view(-1, num_objects, num_objects)
    rel_obj_scores = self.ground_mod(visual, q_rel) 

    # update pair-wise scores by repeating object column-wise and adding spatial scores
    S_rel = rel_obj_scores.unsqueeze(1).repeat(1, num_objects, 1) + rel_scores

    S_pair = S_subj + S_rel + S_loc   # B x N x N

    # return strong/weak supervision output
    return S_pair if self.strong else S_pair.max(-1)[0] 
    
  def strong_bce_loss(self, S_pair: Tensor, gt_subj: Tensor, gt_obj: Tensor) -> Tensor:
    # S_pair: B x N x N, ground_truth_subj: B x N, ground_truth_obj: B x N
    crit = BCEWithLogitsIgnore(ignore_index=-1, reduction='mean')
    raise NotImplementedError

  def weak_bce_loss(self, S_unit: Tensor, gt_subj: Tensor) -> Tensor:
    # S_unit: B x N, ground_truth_subj: B x N
    crit = BCEWithLogitsIgnore(ignore_index=-1, reduction='mean')
    return crit(S_unit, gt_subj)

  def triplet_loss(self, S_unit: Tensor, gt_subj: Tensor) -> Tensor:
    raise NotImplementedError