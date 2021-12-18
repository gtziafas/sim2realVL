from ..types import *
from ..utils.training import Trainer
from ..utils.loss import *
from ..models.vg import *
from ..models.cmn import CMNEndToEnd, Matching

from .prepare_sim_dataset import get_tensorized_dataset

import torch
import torch.nn as nn 
from torchvision.models import resnet18
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW, Adam, SGD
from sklearn.model_selection import KFold
from math import ceil
import yaml

# reproducability
SEED = torch.manual_seed(1312)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(1312)


def main(num_epochs: int,
         model_id: str, 
         pos_emb: str,
         batch_size: int,
         lr: float,
         wd: float,
         device: str,
         console: bool,
         config_path:str, 
         save_path: Maybe[str],
         load_path: Maybe[str],
         checkpoint: Maybe[str],
         early_stopping: Maybe[int],
         onestage: bool,
         kfold: Maybe[int]
         ):

    def train(train_ds: List[AnnotatedScene], dev_ds: List[AnnotatedScene], test_ds: Maybe[List[AnnotatedScene]] = None):
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))
        dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))

        # optionally test in separate split, given from a path directory as argument
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=collate(device)) if test_ds is not None else None

        with open(config_path, 'r') as stream:
            cfg = yaml.safe_load(stream)
        

        if model_id == "Match":
            model =  Matching(cfg['visual_embed_size'],
                              cfg['word_embed_size'],
                              cfg['jemb_size'],
                              cfg['jemb_dropout']
                            ).to(device)
        
        elif model_id != "CMN":
            stage = 1 if onestage else 2
            model = make_model(model_id, pos_emb, stage).to(device)
        
        else:
            model = CMNEndToEnd(cfg).to(device)

        if load_path is not None:
            model.load_pretrained(load_path)

        optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = BCEWithLogitsIgnore(reduction='mean', ignore_index=-1)
        # criterion = TripletHingeLoss()
        
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric="true_positive_rate", early_stopping=early_stopping)
        
        best = trainer.iterate(num_epochs, with_save=save_path, print_log=console)
        return best, trainer.logs 

    # get data
    print('Loading...')
    ds = get_tensorized_dataset(~onestage, "no", False) if not checkpoint else torch.load(checkpoint)
    
    if not kfold:
        # random split
        dev_size, test_size = ceil(len(ds) * .20), 0
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])
        #train_ds, test_ds = random_split(train_ds, [len(train_ds) - test_size, test_size])
        print('Training on random train-dev-test split:')
        best, logs = train(train_ds, dev_ds, None)
        print(f'Results random split: {best}')

    else:
        # k-fold cross validation 
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=1312).split(ds)
        metrics = []
        print(f'{kfold}-fold cross validation...')
        for iteration, (train_idces, dev_idces) in enumerate(_kfold):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            best, logs = train(train_ds, dev_ds, None)
            print(f'Results {kfold}-fold, iteration {iteration+1}: {best}')
            metrics.append(best)
        
        avgs = {}
        for met in metrics:
            for k, v in met.items():
                avgs[k] = v if k not in avgs.keys() else avgs[k] + v
        print(f'Averaged results {kfold}-fold:')
        for k, v in avgs.items():
            print(f'\t{k}: {round(v / kfold, 4)}')
    # print([i['loss'] for i in logs['train']])
    # print([i['loss'] for i in logs['dev']]) 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=50)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=10)
    parser.add_argument('-f', '--model_id', help='what type of fusion module to use (MLP, RNN)', type=str, default="MLP")
    parser.add_argument('-pe', '--pos_emb', help='what type of positional embeddings to use (no, raw, harmonic)', type=str, default="raw")
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=None)
    parser.add_argument('-l', '--load_path', help='where to load model from', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-cfg', '--config_path', help='path to configuration file', type=str, default='./sim2realVL/configs/cmn.yaml')
    #parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.25)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=1e-03)
    parser.add_argument('--console', action='store_true', help='print training logs', default=False)
    parser.add_argument('--onestage', action='store_true', help='whether to train one-stage model varianet', default=False)
    parser.add_argument('-chp', '--checkpoint', help='load dataset from binary file', default=None)
    parser.add_argument('-kfold', '--kfold', help='whether to do k-fold x-validation (default no)', type=int, default=None)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)