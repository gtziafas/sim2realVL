from ..types import *
from ..utils.training import Trainer
from ..utils.metrics import Metrics, vg_metrics
from ..data.rgbd_scenes import RGBDScenesVG
from ..models.vg import *

import torch
import torch.nn as nn 
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW, Adam, SGD
from sklearn.model_selection import KFold
from math import ceil

# reproducability
SEED = torch.manual_seed(1312)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(1312)


def main(num_epochs: int,
         batch_size: int,
         lr: float,
         wd: float,
         device: str,
         print_log: bool,
         save_path: Maybe[str],
         load_path: Maybe[str],
         checkpoint: Maybe[str],
         early_stopping: Maybe[int],
         kfold: Maybe[int]
         ):
    def train(train_ds: List[AnnotatedScene], dev_ds: List[AnnotatedScene], test_ds: Maybe[List[AnnotatedScene]] = None) -> Metrics:
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))
        dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))

        # optionally test in separate split, given from a path directory as argument
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=collate(device)) if test_ds is not None else None

        #model = default_vg_model().to(device)
        #model = fast_vg_model().to(device)
        model = MultiLabelRNNVG(visual_encoder=None, text_encoder=RNNContext(300, 150, 1), 
                                fusion_dim=200, num_fusion_layers=1, with_downsample=True).to(device)
        #model = MultiLabelMHAVG(visual_encoder=None, fusion_dim=300, num_heads=3).to(device)
        if load_path is not None:
            model.load_pretrained(load_path)
        optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, metrics_fn = vg_metrics,
                target_metric="accuracy", early_stopping=early_stopping)
        
        return trainer.iterate(num_epochs, with_save=save_path, print_log=print_log)

    # get data
    print('Loading...')
    ds = make_vg_dataset(RGBDScenesVG()) if not checkpoint else torch.load(checkpoint)
    
    if not kfold:
        # random split
        dev_size, test_size = ceil(len(ds) * .1), ceil(len(ds) * .15)
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])
        train_ds, test_ds = random_split(train_ds, [len(train_ds) - test_size, test_size])
        print('Training on random train-dev-test split:')
        best = train(train_ds, dev_ds, test_ds)
        print(f'Results random split: {best}')

    else:
        # k-fold cross validation 
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=14).split(ds)
        accu = 0.
        print(f'{kfold}-fold cross validation...')
        for iteration, (train_idces, dev_idces) in enumerate(_kfold):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            best = train(train_ds, dev_ds, None)
            print(f'Results {kfold}-fold, iteration {iteration+1}: {best}')
            accu += best['accuracy']
        print(f'Average accuracy {kfold}-fold: {accu/kfold}')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=64)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=10)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=None)
    parser.add_argument('-l', '--load_path', help='where to load model from', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    #parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.25)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=1e-03)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('-chp', '--checkpoint', help='load pre-trained visual features from file', default=None)
    parser.add_argument('-kfold', '--kfold', help='whether to do k-fold x-validation (default no)', type=int, default=None)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)