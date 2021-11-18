from ..types import *
from ..utils.image_proc import crop_box, crop_contour
from ..data.sim_dataset import get_sim_rgbd_scenes_annotated
from ..utils.training import Trainer
from ..utils.word_embedder import make_word_embedder
from ..utils.loss import BCEWithLogitsIgnore
from ..models.visual_embedder import make_visual_embedder
from ..models.vg import *

import torch
import torch.nn as nn 
from torchvision.models import resnet18
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW, Adam, SGD
from sklearn.model_selection import KFold
from math import ceil
from tqdm import tqdm

# reproducability
SEED = torch.manual_seed(1312)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(1312)


def prepare_dataset(ds: List[AnnotatedScene], 
                    image_loader: Callable[[int], array],
                    pretrained_features: bool, 
                    save: Maybe[str] = None
                    ):
    H, W = 480, 640
    we = make_word_embedder()
    ve = make_visual_embedder()

    dataset = []
    covered_ids = {}
    for i, scene in enumerate(tqdm(ds)):
        # dont do rendundant cropping
        if scene.image_id not in covered_ids:
            crops = [crop_contour(image_loader(scene.image_id), o.contour) for o in scene.objects]
            feats = ve.features(crops) if pretrained_features else torch.stack(ve.tensorize(crops))
            covered_ids[scene.image_id] = feats
        else:
            feats = covered_ids[scene.image_id]
        
        truth = torch.zeros(len(scene.objects), dtype=longt)
        truth[scene.truth] = 1
        position = torch.stack([torch.tensor([b.x/W, b.y/H, (b.x+b.w)/W, (b.y+b.h)/H]).float() for b in scene.boxes])
        
        query = torch.tensor(we([scene.query])[0], dtype=floatt)
        
        dataset.append((feats, query, truth, position))

    if save is not None:
        torch.save(dataset, save)

    return dataset


def main(num_epochs: int,
         model_id: str, 
         pos_emb: str,
         batch_size: int,
         lr: float,
         wd: float,
         device: str,
         console: bool,
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

        stage = 1 if onestage else 2  
        model = make_model(stage, model_id, pos_emb).to(device)
        if load_path is not None:
            model.load_pretrained(load_path)

        optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
        #optim = SGD(model.parameters(), lr=lr, momentum=.9)
        #criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        criterion = BCEWithLogitsIgnore(reduction='mean', ignore_index=-1)
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric="f1_score", early_stopping=early_stopping)
        
        best = trainer.iterate(num_epochs, with_save=save_path, print_log=console)
        return best, trainer.logs 

    # get data
    print('Loading...')
    ds = get_sim_rgbd_scenes_annotated()
    ds = prepare_dataset(prepare_dataset(ds, ds.get_image_from_id, not onestage)) if not checkpoint else torch.load(checkpoint)
    
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
        print(metrics)
    # print([i['loss'] for i in logs['train']])
    # print([i['loss'] for i in logs['dev']]) 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=64)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=10)
    parser.add_argument('-f', '--model_id', help='what type of fusion module to use (MLP, RNN)', type=str, default="MLP")
    parser.add_argument('-pe', '--pos_emb', help='what type of positional embeddings to use (no, raw, harmonic)', type=str, default="raw")
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=None)
    parser.add_argument('-l', '--load_path', help='where to load model from', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    #parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.25)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=1e-03)
    parser.add_argument('--console', action='store_true', help='print training logs', default=False)
    parser.add_argument('--onestage', action='store_true', help='whether to train one-stage model varianet', default=False)
    parser.add_argument('-chp', '--checkpoint', help='load pre-trained visual features from file', default=None)
    parser.add_argument('-kfold', '--kfold', help='whether to do k-fold x-validation (default no)', type=int, default=None)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)