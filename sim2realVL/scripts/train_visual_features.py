from ..types import *
from ..models.visual_embedder import *
from ..data.sim_dataset import get_sim_rgbd_objects
from ..utils.image_proc import crop_boxes_fixed
from ..utils.training import Trainer, train_epoch_classifier, eval_epoch_classifier

import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from math import ceil

# set gpu if available
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

# reproducability
SEED = torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)

# hard-code image size
IMG_SIZE = (75, 75)


def train_classifier(ne: int, bs: int, lr: float, wd: float, early: int, device: str, console: bool, save: Maybe[str]):
    # load default model
    model = custom_classifier().to(device)

    # load and preprocess dataset
    dataset = get_sim_rgbd_objects()
    crops, labels = zip(*dataset)
    crops = crop_boxes_fixed(IMG_SIZE)(list(crops))
    dataset = list(zip(crops, labels))
    labelset = {v: k for k, v in enumerate(sorted(set(labels)))}

    # 80-20 random split
    test_size = ceil(.20 * len(dataset))
    train_ds, test_ds = random_split(dataset, [len(dataset) - test_size, test_size], generator=SEED)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs, collate_fn=collate(device, labelset), worker_init_fn=SEED)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=bs, collate_fn=collate(device, labelset), worker_init_fn=SEED)

    # init optimizer and objective
    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # train
    trainer = Trainer(model, (train_dl, test_dl, None), optim, criterion, eval_fn=eval_epoch_classifier,
                target_metric="accuracy", early_stopping=early, train_fn=train_epoch_classifier)
    print("Training...")
    best = trainer.iterate(ne, with_save=save, print_log=console)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--bs', help='batch size to use for training', type=int, default=64)
    parser.add_argument('-e', '--ne', help='how many epochs of training', type=int, default=10)
    parser.add_argument('-s', '--save', help='where to save best model', type=str, default=None)
    #parser.add_argument('-l', '--load_path', help='where to load model from', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.0)
    #parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.25)
    parser.add_argument('-early', '--early', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=1e-03)
    parser.add_argument('--console', action='store_true', help='print training logs', default=False)
    
    kwargs = vars(parser.parse_args())
    train_classifier(**kwargs)