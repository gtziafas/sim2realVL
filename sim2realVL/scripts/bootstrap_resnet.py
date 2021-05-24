from ..types import * 
from ..data.rgbd_scenes import RGBDScenesVG

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchvision import transforms as T 
from torchvision.models import resnet18
from PIL import Image
from math import ceil

seed = torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)

tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

LABELS = ['bowl_2',
 'bowl_3',
 'bowl_4',
 'cap_1',
 'cap_3',
 'cap_4',
 'cereal_box_1',
 'cereal_box_2',
 'cereal_box_4',
 'coffee_mug_1',
 'coffee_mug_4',
 'coffee_mug_5',
 'coffee_mug_6',
 'flashlight_1',
 'flashlight_2',
 'flashlight_3',
 'flashlight_5',
 'soda_can_1',
 'soda_can_3',
 'soda_can_4',
 'soda_can_5',
 'soda_can_6']


def collate(device: str = 'cuda', supervised: bool = True) -> Map[Sequence[Object], Tuple[Tensor, MayTensor]]:
    def _collate(batch: Sequence[ObjectCrop]) -> Tuple[Tensor, MayTensor]:
        imgs, labels = zip(*[(s.image, s.label) for s in batch])
        imgs = [Image.fromarray(img) for img in imgs]
        imgs = list(map(tf, imgs))
        imgs = torch.stack(imgs).to(device)
        _labels = None
        if supervised:
            _labels = torch.stack([torch.tensor(LABELS.index(l), dtype=longt, device=device) for l in labels])
        return imgs, _labels
    return _collate


def train_epoch(model, train_dl, optim, criterion):
    model.train()
    epoch_loss, total_correct = 0., 0
    for x, y in train_dl:
        preds = model(x)
        loss = criterion(preds, y)
        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(train_dl)
    total_correct /= len(train_dl.dataset) 
    return epoch_loss, total_correct


@torch.no_grad()
def eval_epoch(model, dev_dl, criterion):
    model.eval()
    epoch_loss, total_correct = 0., 0.
    for x, y in dev_dl:
        preds = model(x)
        loss = criterion(preds, y)
        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(dev_dl)
    total_correct /= len(dev_dl.dataset) 
    return epoch_loss, total_correct


def train(ne: int, bs: int, lr: float, wd: float, dr: float, pretrained: bool, save: Maybe[str] = None):
    # get data
    print('Fetching data...')
    ds = RGBDObjectsDataset()
    dev_size, test_size = ceil(.1 * len(ds)), ceil(.1 * len(ds))
    train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size], generator=seed)
    train_ds, test_ds = random_split(train_ds, [len(train_ds) - test_size, test_size], generator=seed)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs, collate_fn=collate(), worker_init_fn=seed)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=bs, collate_fn=collate(), worker_init_fn=seed)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=bs, collate_fn=collate(), worker_init_fn=seed)

    # init model, optim, loss
    print('Model init...')
    model = resnet18(pretrained=pretrained).cuda()
    model.fc = torch.nn.Sequential(torch.nn.Dropout(dr), torch.nn.Linear(512, 22)).cuda()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    print('Training...')
    best = {'accuracy' : 0.}; latest_test = 0.
    for epoch in range(ne):
        # training
        loss, accu = train_epoch(model, train_dl, optim, criterion)
        print(f'TRAIN {epoch+1}/{ne}: loss={loss:.4f}, accu={100*accu:2.2f}%')

        # evaluation
        loss, accu = eval_epoch(model, dev_dl, criterion)
        print(f'DEV {epoch+1}/{ne}: loss={loss:.4f}, accu={100*accu:2.2f}%')

        # selection
        if accu > best['accuracy']:
            best = {'epoch': epoch+1, 'loss': loss, 'accuracy': accu}
            test_loss, test_accu = eval_epoch(model, test_dl, criterion)
            print(f'====TEST====: loss = {test_loss:.4f}, accu={100*test_accu:2.2f}%')
            if save is not None:
                torch.save(model.state_dict(), save)    


@torch.no_grad()
def extract_features_vg(save_path: str, load_path: str):
    # get data
    print('Fetching data...')
    ds = RGBDScenesVG()
        
    # get model 
    print('Loading model...')
    model = resnet18().cuda()
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.), torch.nn.Linear(512, 22)).cuda()
    model.load_state_dict(torch.load(load_path))
    model.fc = torch.nn.Identity().cuda()
    model.eval()

    print('Computing all features...')
    idces = [[i for i, p in enumerate(ds.rgb_paths) if p == path] for path in ds.unique_scenes]
    all_feats = []
    for ids in idces:
        scene = ds[ids[0]]
        crops = scene.get_crops()
        crops = torch.stack([tf(Image.fromarray(c)) for c in crops]).cuda()    # N x 3 x H x W
        all_feats.append(model(crops))    # N x 512

    print(f'Saving in {save_path}...')
    torch.save(all_feats, save_path)
