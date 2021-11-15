from ..types import * 
from ..utils.image_proc import crop_boxes_fixed
from ..data.rgbd_scenes import RGBDObjectsDataset

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam
from torchvision import transforms as T 
from PIL import Image
from math import ceil
from ..data.sim_dataset import get_sim_rgbd_scenes_old

seed = torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)

_SIZE = (120, 120)
tf = T.Compose([T.CenterCrop(224), T.Resize(120), T.ToTensor()]) 
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

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

SIM_LABELS = ['mug_red', 'can_pepsi', 'flashlight_black', 'flashlight_yellow', 'flashlight_red', 'can_fanta', 'cereal_box_2', 'cereal_box_3', 'flashlight_blue', 'cap_black', 'bowl_1', 'can_coke', 'cap_white', 'cereal_box_1', 'mug_yellow', 'can_sprite', 'cap_red', 'mug_green', 'bowl_2']


def collate_empty(device: str = 'cuda'):
    def _collate(batch):
        imgs ,labels = zip(*batch)
        imgs = torch.stack(imgs).view(-1, 3, *_SIZE)
        labels = torch.stack(labels)
        return imgs.to(device), labels.to(device)
    return _collate 


def collate(device: str = 'cuda', supervised: bool = True) -> Map[Sequence[Object], Tuple[Tensor, MayTensor]]:
    def _collate(batch: Sequence[ObjectCrop]) -> Tuple[Tensor, MayTensor]:
        imgs, labels = zip(*[(s.image, s.label) for s in batch])
        imgs = crop_boxes_fixed(_SIZE)(list(imgs))
        imgs = [torch.tensor(img, dtype=floatt, device=device).div(0xff) for img in imgs]
        imgs = torch.stack(imgs).view(-1, 3, *_SIZE)
        _labels = None
        if supervised:
            _labels = torch.stack([torch.tensor(LABELS.index(l), dtype=longt, device=device) for l in labels])
        return imgs.to(device), _labels.to(device)
    return _collate


def collate_other(device="cuda"):
    def _collate(batch):
        imgs, labels = zip(*batch)
        imgs = crop_boxes_fixed(_SIZE)(list(imgs))
        imgs = [torch.tensor(img, dtype=floatt, device=device).div(0xff) for img in imgs]
        imgs = torch.stack(imgs).view(-1, 3, *_SIZE)
        labels = torch.stack([torch.tensor(SIM_LABELS.index(l), dtype=longt, device=device) for l in labels])
        return imgs, labels
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


def train(model: nn.Module, ne: int, bs: int, lr: float, wd: float, dr: float, pretrained: bool, save: Maybe[str] = None):
    # get data
    print('Fetching data...')
    #ds = RGBDObjectsDataset()
    # import pickle
    # ds = pickle.load(open("checkpoints/sim_for_resnet.p", "rb"))
    ds = torch.load("checkpoints/TORA_sim_objects_dataset.p")
    dev_size, test_size = ceil(.15 * len(ds)), ceil(.05 * len(ds))
    train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size], generator=seed)
    train_ds, test_ds = random_split(train_ds, [len(train_ds) - test_size, test_size], generator=seed)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs, collate_fn=collate_empty(), worker_init_fn=seed)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=bs, collate_fn=collate_empty(), worker_init_fn=seed)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=bs, collate_fn=collate_empty(), worker_init_fn=seed)

    # init model, optim, loss
    print('Model init...')
    model = model(pretrained=pretrained).cuda()
    model.fc = torch.nn.Sequential(torch.nn.Dropout(dr), torch.nn.Linear(512, len(SIM_LABELS))).cuda()
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
def extract_features_vg(model: nn.Module, load_path: Maybe[str], save_path: Maybe[str] = None) -> List[Tensor]:
    # get data
    from ..data.rgbd_scenes import RGBDScenesVG
    print('Fetching data...')
    ds = RGBDScenesVG()
        
    # get model 
    print('Loading model...')
    model = model().cuda()
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.), torch.nn.Linear(512, len(SIM_LABELS))).cuda()
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model.fc = torch.nn.Identity().cuda()
    model.eval()

    print('Computing all features...')
    idces = [[i for i, p in enumerate(ds.rgb_paths) if p == path] for path in ds.unique_scenes]
    all_feats = []
    for ids in idces:
        scene = ds[ids[0]]
        crops = crop_boxes_fixed((120, 120))(scene.get_crops())
        crops = torch.stack([torch.tensor(c, dtype=floatt, device='cuda').div(0xff) for c in crops]).view(-1, 3, 120, 120)
        feats = model(crops) 
        for i in range(feats.shape[0]):
            all_feats.append(feats)
    if save_path is not None:
        print(f'Saving in {save_path}...')
        torch.save(all_feats, save_path)
    return all_feats


@torch.no_grad()
def extract_features_sim(ds: List[AnnotatedScene], load_resnet: str, save_path: str):
    ds = [s for s in ds]
    from torchvision.models import resnet18
    resnet = resnet18()
    resnet.fc = torch.nn.Sequential(torch.nn.Dropout(0.), torch.nn.Linear(512, len(SIM_LABELS))).cuda()
    resnet.load_state_dict(torch.load(load_resnet))
    resnet.fc = torch.nn.Identity()
    resnet = resnet.eval().cpu()

    all_feats = []
    for scene in ds:
        crops = crop_boxes_fixed((120, 120))(scene.get_crops())
        crops = torch.stack([torch.tensor(c, dtype=floatt).div(0xff) for c in crops]).view(-1, 3, 120, 120)
        feats = resnet(crops) 
        for i in range(feats.shape[0]):
            all_feats.append(feats)
    if save_path is not None:
        torch.save(all_feats, save_path)
    return all_feats


from torchvision.models import mobilenet_v3_small, resnet18
#train(resnet18, 10, 32, 1e-03, 0, 0.25, False, 'checkpoints/TORA_resnet18_sim.p')
extract_features_sim(get_sim_rgbd_scenes_old(), 'checkpoints/TORA_resnet18_sim.p', 'checkpoints/TORA_sim_visual_features.p')