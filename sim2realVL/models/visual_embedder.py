from ..types import *
from ..utils.image_proc import *
from .nets import MLP, CNNFeatures, CNNClassifier

import torch
import torch.nn as nn 

DEFAULT_CNN_PARAMS ={"num_blocks": 3, 
              "conv_kernels": [3, 3, 3],
              "dropout_rates": [0, 0.05, 0.25],
              "pool_kernels": [3, 3, 2],
              }


class VisualEmbedder(ABC):
    def __init__(self, 
                 feature_extractor: nn.Module,
                 preprocess: Map[List[array], List[array]],
                 device: str
                 ):
        self.preprocess = preprocess
        self.device = device 
        self.feature_extractor = feature_extractor.eval().to(device)

    def _tensorize(self, x: array) -> Tensor:
        x = torch.tensor(x/0xff, dtype=torch.float, device=self.device)
        return x.view(x.shape[-1], *x.shape[0:2])

    def tensorize(self, xs: List[array]) -> List[Tensor]:
        xs = self.preprocess(xs)
        return list(map(self._tensorize, xs))

    @torch.no_grad()
    def features(self, xs: List[array]) -> Tensor:
        xs = torch.stack(self.tensorize(xs), dim=0)
        feats = self.feature_extractor(xs).flatten(1)
        return feats

    def load_weights(self, path: str):
        try:
            self.feature_extractor.load_state_dict(torch.load(path))
        except: 
            model = custom_classifier()
            model.load_state_dict(torch.load(path))
            self.feature_extractor = model.features.eval().to(self.device)


def custom_classifier(num_classes: int = 19, num_features: int = 256) -> nn.Module:
    return CNNClassifier(CNNFeatures(**DEFAULT_CNN_PARAMS), nn.Linear(num_features, num_classes))


def custom_features() -> nn.Module:
    return CNNFeatures(**DEFAULT_CNN_PARAMS)


def resnet_classifier(load: Maybe[str] = None, num_classes: int = 19, dropout: float = 0.5, pretrained: bool = False) -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def resnet_features(load: Maybe[str] = None, pretrained: bool = False) -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Identity()
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def alexnet_classifier(load: Maybe[str] = None, num_classes: int = 19, dropout: float = 0.5, pretrained: bool = False) -> nn.Module:
    from torchvision.models import alexnet
    model = alexnet(pretrained=pretrained)
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(4096, num_classes))
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def alexnet_features(load: Maybe[str] = None, num_classes: int = 19, num_features: int = 512, pretrained: bool = False):
    from torchvision.models import alexnet
    model = resnet18(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(4096, num_features)
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def _tensorize(x: array) -> Tensor:
    x = torch.tensor(x/0xff, dtype=torch.float)
    return x.view(x.shape[-1], *x.shape[0:2])


def tensorize(xs: List[array]) -> Tensor:
    return torch.stack(list(map(_tensorize, xs)), dim=0)


# assuming images already pre-processed
def collate(device: str, labelset: Set[str]) -> Map[List[Tuple[array, str]], Tuple[Tensor, Tensor]]:
    idces = {v: k for k, v in enumerate(labelset)}
    def _collate(batch: List[Tuple[array, str]]) -> Tuple[Tensor, Tensor]:
        xs, ys = zip(*batch)
        xs = tensorize(xs).to(device)
        ys = torch.stack([torch.tensor(idces[y], dtype=longt, device=device) for y in ys], dim=0)
        return xs, ys
    return _collate


def make_visual_embedder(load: str = "./checkpoints/SIM_classifier.p", device = "cpu"):
    ve = VisualEmbedder(feature_extractor=custom_features(),
                        preprocess=crop_boxes_fixed((75, 75)),
                        device=device)
    ve.load_weights(load)
    print('Loaded pre-trained visual embedder...')
    return ve 


def extract_features_from_dataset(dataset: List[Scene], 
                                  image_loader: Callable[[int], array],
                                  save: Maybe[str] = None) -> List[array]:
    VE = make_visual_embedder()
    feats = []
    for i, scene in enumerate(dataset):
        img = image_loader(i)
        crops = [crop_box(img, o.box) for o in scene.objects]
        feats.append(VE.features(crops))
    if save is not None:
        torch.save(feats, save)
    return feats