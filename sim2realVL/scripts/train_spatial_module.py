from ..types import *
from ..data.sim_dataset import *
from ..models.position_embedder import make_position_embedder
from ..models.word_embedder import make_word_embedder
from ..models.cmn import WordTagger, SpatialRelations
from ..utils.loss import BCEWithLogitsIgnore
from ..data.scene_parser import SceneParser 
from ..utils.metrics import *

import torch
import torch.nn as nn 
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from tqdm import tqdm
import yaml


class SpatialModule(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.position_embedder = make_position_embedder()
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
                              ).apply(self.init_weights)

        self.spt_rel = SpatialRelations(cfg['position_embed_size'],
                              cfg['word_embed_size'],
                              cfg['jemb_size'],
                              cfg['jemb_dropout']
                              ).apply(self.init_weights)
        
        self.padding_value = cfg['padding_value']
        if cfg['word_tagger_pretrain'] is not None:
            self.word_tagger.load_state_dict(torch.load(cfg['word_tagger_pretrain']))
            for param in self.word_tagger.parameters():
                param.requires_grad = False

    def init_weights(self, net: nn.Module):
        if isinstance(net, nn.Linear):
            nn.init.kaiming_uniform_(net.weight)
            net.bias.data.fill_(0.01)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        # position: B x N x Dp, query: B x T x Dt
        queries, position = inputs
        position = self.position_embedder(position)
        batch_size, num_objects = position.shape[0:2]

        # N x N pair-wise position embeddings, flattened
        p_tile = position.unsqueeze(2).repeat(1, 1, num_objects, 1)
        p_pair = torch.cat([p_tile, p_tile.transpose(1,2)], dim=-1)  
        p_pair = p_pair.view(batch_size, num_objects**2, -1) # B x N^2 x 2*Dp

        # average word embeddigns according to predicted tag
        pad_mask = (queries == self.padding_value).sum(-1) == queries.shape[-1]
        (q_loc, _, q_rel, _), q_context = self.word_tagger.encode_per_tag(queries, pad_mask) # (4x) B x Dt
        
        # merge loc + rel 
        assert (q_loc.sum(1) * q_rel.sum(1)).sum() == 0 
        q_spt = q_loc + q_rel 
        scores = self.spt_rel(p_pair, q_spt).view(batch_size, num_objects, num_objects) # B x N x N 
        
        return scores


class SpatialModuleTest(object):
    def __init__(self, 
                 load: str = "./checkpoints/cmn/SIM_spatial_pairwise.p"):
        config_path =  './sim2realVL/configs/cmn.yaml'
        with open(config_path, 'r') as stream:
            cfg = yaml.safe_load(stream)
        self.net = SpatialModule(cfg)
        self.net.spt_rel.load_state_dict(torch.load(load))
        self.net = self.net.spt_rel.eval()
        self.PE = make_position_embedder()
        self.WE = make_word_embedder()

    @torch.no_grad()
    def pairwise_matrix(self, positions: List[array], query: str) -> array:
        N = len(positions)
        q_spt = torch.tensor(self.WE([query])[0]).mean(0, keepdim=True)

        pos_embs = self.PE(torch.stack([torch.tensor(p) for p in positions]).unsqueeze(0)) # 1 x N x 8
        p_tile = pos_embs.unsqueeze(2).repeat(1, 1, N, 1)
        p_pair = torch.cat([p_tile, p_tile.transpose(1,2)], dim=-1)  
        p_pair = p_pair.view(1, N**2, -1) # 1 x N^2 x 8
        
        matrix = self.net(p_pair, q_spt).squeeze().view(N, N).sigmoid().ge(0.5).cpu().numpy()
        return matrix


def extract_spatial_masks(ds: List[AnnotatedScene]):
    keywords = ['left', 'right', 'behind', 'front', 'closest', 'furthest', 'next']
    parser = SceneParser()
    masks = []
    for scene in tqdm(ds):
        graph = parser(scene)
        query = scene.query
        
        for key in keywords:
            if key in query:
                masks.append(graph.get_mask(key))
                break
    return masks

    
def make_spatial_dataset(load_chp: str, save_chp: str):
    ds = get_sim_rgbd_scenes_annotated()
    exclude_ids = get_split_indices(ds, "visual")
    ds = [s for i, s in enumerate(ds) if i not in exclude_ids]

    masks = extract_spatial_masks(ds)
    masks = [torch.tensor(m, dtype=longt) for m in masks]

    ds = torch.load(load_chp)
    ds = [s for i, s in enumerate(ds) if i not in exclude_ids]
    _, Qs, _, Ps = zip(*ds)

    assert len(masks) == len(Qs) == len(Ps)
    assert len(set([p.shape[0] for p in Ps]).difference(set([m.shape[0] for m in masks]))) == 0 
    
    torch.save(list(zip(Qs, Ps, masks)), save_chp)


def collate(device: str, padding_value: int = - 1) -> Map[List[Tuple[Tensor,...]], Tuple[Tensor, ...]]:
    def _collate(batch: List[Tuple[Tensor,...]]) -> Tuple[Tensor, ...]:
        qs, ps, masks = zip(*batch)
        qs = pad_sequence(qs, batch_first=True, padding_value=padding_value)
        ps = pad_sequence(ps, batch_first=True, padding_value=padding_value)
        # manual padding 
        maxlen = ps.shape[1]
        masks_padded = padding_value * torch.ones((len(batch), maxlen, maxlen), dtype=floatt)
        for i, m in enumerate(masks):
            masks_padded[i, :m.shape[0], :m.shape[0]] = m
        return qs.to(device), ps.to(device), masks_padded.to(device)
    return _collate


def train_epoch(model: nn.Module, dl: DataLoader, optim: Optimizer, criterion: nn.Module) -> Metrics:
    model.train()
    
    epoch_loss = 0
    confusion_matrix = torch.zeros(3, 2)
    for batch_idx, (words, positions, masks) in enumerate(dl):
        preds = model.forward([words, positions])
        loss = criterion(preds, masks.float())

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        # metrics
        epoch_loss += loss.item()
        confusion_matrix += get_confusion_matrix(preds.sigmoid().ge(0.5), masks)

    epoch_loss /= len(dl)
    return {'loss': -round(epoch_loss, 5), **get_metrics_from_matrix(confusion_matrix)}


@torch.no_grad()
def eval_epoch(model: nn.Module, dl: DataLoader, criterion: nn.Module) -> Metrics:
    model.eval()

    confusion_matrix = torch.zeros(3, 2)
    epoch_loss = 0
    for batch_idx, (words, positions, masks) in enumerate(dl):
        preds = model.forward([words, positions])
        loss = criterion(preds, masks.float())
        epoch_loss += loss.item()
        confusion_matrix += get_confusion_matrix(preds.sigmoid().ge(0.5), masks)

    epoch_loss /= len(dl)
    return {'loss': -round(epoch_loss, 5), **get_metrics_from_matrix(confusion_matrix)}


def main():
    ds = torch.load('./checkpoints/cmn/SIM_dataset_spatial_module.p')

    dev_size = ceil(len(ds) * .20)
    train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])

    batch_size = 5
    train_dl = DataLoader(ds, shuffle=True, batch_size=batch_size, collate_fn=collate("cuda"))
    dev_dl = DataLoader(ds, shuffle=False, batch_size=batch_size, collate_fn=collate("cuda"))

    config_path =  './sim2realVL/configs/cmn.yaml'
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)
    
    model = SpatialModule(cfg).cuda()
    crit = BCEWithLogitsIgnore(ignore_index=-1, reduction='mean').cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-03, weight_decay=0.)

    num_epochs = 2
    print(f'Training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_dl, opt, crit)
        test_metrics = eval_epoch(model, dev_dl, crit)
        print(f'Iteration={epoch+1}:')
        print(train_metrics)
        print()
        print()
        print(test_metrics)
        print('==' * 48)

    torch.save(model.spt_rel.state_dict(), './checkpoints/cmn/SIM_spatial_pairwise.p')


if __name__ == "__main__":
  main()
