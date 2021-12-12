from ..types import *
from ..data.sim_dataset import *
from ..utils.word_embedder import make_word_embedder
from ..models.cmn import WordTagger 
from ..utils.loss import BCEWithLogitsIgnore

import torch
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from math import ceil


def annotate_word_tags(ds: List[AnnotatedScene]):
    tagset = ['<loc>', '<obj>', '<rel>', '<subj>']
    colors = ['blue', 'green', 'white', 'black', 'red', 'purple', 'brown', 'orange', 'yellow']
    all_subj_words = set(sum([s.categories for s in ds], [])) 
    all_subj_words = sum([c.split() for c in all_subj_words], []) + colors

    def transform(split: str) -> Map[str, List[str]]:
        if split == "category" or split == "color":
            return lambda q: ['<subj>'] * len(q.split()) 

        elif split == "spatial_abs":
            return lambda q: ['<loc>'] + ['<subj>'] * (len(q.split()) - 1)

        elif split == "spatial_rel":
            def _transform_spt_rel(q: str) -> List[str]:
                tokens = q.split()
                is_rel = [0 if t in all_subj_words else 1 for t in tokens]
                rel_starts, rel_ends = is_rel.index(1), (len(is_rel) - is_rel[::-1].index(1))
                return ['<subj>'] * rel_starts + ['<rel>'] * (rel_ends - rel_starts) + ['<obj>'] * (len(is_rel) - rel_ends) 
            return _transform_spt_rel

        else:
            raise ValueError(f"unknown split {split}")
            
    queries = ds.queries
    annots = [None] * len(queries)
    for split in ["category", "color", "spatial_abs", "spatial_rel"]:
        tmp_indices = get_split_indices(ds, split)
        for index in tmp_indices:
            annots[index] = split
    assert None not in annots

    return [transform(annot)(query) for query, annot in zip(queries, annots)], tagset


def get_dataset_word_tags():
    ds = get_sim_rgbd_scenes_annotated()
    return annotate_word_tags(ds)


def get_dataloaders_pretrain(queries: List[str], 
                             tags: List[str], 
                             batch_size: int, 
                             device: str,
                             tagset: List[str],
                             load_checkpoint: Maybe[str] = None,
                             save_checkpoint: Maybe[str] = None):

  if load_checkpoint is None:
    from sim2realVL.models.word_embedder import make_word_embedder
    WE = make_word_embedder()
    word_vectors = WE(queries) # list of arrays
    word_vectors = [torch.tensor(v) for v in word_vectors]

    # define tag vocabulary and tokenizer
    tag_vocab = {k: v for k,v in enumerate(tagset)}
    tag_vocab_inv = {v : k for k, v in tag_vocab.items()}
    
    # tokenize and convert to one-hot
    tag_ids = [torch.tensor([tag_vocab_inv[t] for t in ts], dtype=longt) for ts in tags]
    tag_vectors = [torch.eye(len(tag_vocab))[t] for t in tag_ids]

    if save_checkpoint is not None:
      torch.save(list(zip(word_vectors, tag_vectors)), save_checkpoint)

    dataset = list(zip(word_vectors, tag_vectors))

  else:
      dataset = torch.load(load_checkpoint)
      
  def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Map[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
    words, tags = zip(*batch)
    words = pad_sequence(words, batch_first=True, padding_value=-1)
    tags = pad_sequence(tags, batch_first=True, padding_value=-1)
    return words.to(device), tags.argmax(-1).to(device)

  dev_size = ceil(len(queries) * .20)
  train_ds, dev_ds = random_split(dataset, [len(queries) - dev_size, dev_size])

  train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
  dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
  
  return train_dl, dev_dl


@torch.no_grad()
def accuracy_metrics(predictions: Tensor, truth: Tensor, ignore_idx: int = -1) -> Tuple[int, ...]:
    num_sents_total = predictions.shape[0]
    num_items_total = predictions.shape[0] * predictions.shape[1]

    correct_items = torch.ones_like(predictions)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sents = correct_items.prod(dim=1)
    num_correct_sents = correct_sents.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truth[truth == ignore_idx])

    return num_sents_total, num_correct_sents, num_items_total - num_masked_items, num_correct_items - num_masked_items


def main():
    ds = get_sim_rgbd_scenes_annotated()
    queries = ds.queries 
    tags, tagset = annotate_word_tags(ds)

    train_dl, dev_dl = get_dataloaders_pretrain(queries, tags, 128, "cuda", tagset=tagset,
        save_checkpoint="./checkpoints/cmn/SIM_dataset_queries_and_tags.p")

    model = WordTagger(len(tagset), 300, 150, 1, pretrain=True).cuda()
    crit = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=0.)

    num_epochs = 5
    print('Training for {} epochs'.format(num_epochs))
    for epoch in range(num_epochs):
        train_loss = 0. 
        num_sents, num_correct_sents, num_words, num_correct_words = 0,0,0,0
        model.train() 
        opt.zero_grad()
        for x, y in train_dl:
            preds, _ = model(x)
            loss = crit(preds.view(-1, len(tagset)), y.flatten())
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            metrics = accuracy_metrics(preds.argmax(-1), y)
            num_sents += metrics[0]
            num_correct_sents += metrics[1]
            num_words += metrics[2] 
            num_correct_words += metrics[3]

        train_loss /= len(train_dl)
        train_word_accu = num_correct_words / num_words
        train_sent_accu = num_correct_sents / num_sents 

        # testing 
        test_loss = 0. 
        num_sents, num_correct_sents, num_words, num_correct_words = 0,0,0,0
        model.eval()
        for x, y in train_dl:
            with torch.no_grad():
                preds, _ = model(x)
                loss = crit(preds.view(-1, len(tagset)), y.flatten())
            
            test_loss += loss.item()
            metrics = accuracy_metrics(preds.argmax(-1), y)
            num_sents += metrics[0]
            num_correct_sents += metrics[1]
            num_words += metrics[2] 
            num_correct_words += metrics[3]

        test_loss /= len(train_dl)
        test_word_accu = num_correct_words / num_words
        test_sent_accu = num_correct_sents / num_sents

        print('Epoch={}, train: {:.5f} {:.4f} {:.4f} \t test: {:.5f} {:.4f} {:.4f}'.format(epoch+1, 
          train_loss, train_word_accu, train_sent_accu, test_loss, test_word_accu, test_sent_accu))

    print('Saving model')
    torch.save(model.state_dict(), "./checkpoints/cmn/SIM_word_tagger.p")


if __name__ == "__main__":
  main()