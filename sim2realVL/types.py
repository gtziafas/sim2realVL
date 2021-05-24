from typing import List, Tuple, Dict, Any, Callable, Set, TypeVar, Sequence, Iterator
from typing import Optional as Maybe
from typing import Mapping as Map 
from dataclasses import dataclass 
from abc import ABC 

from numpy import array
from torch import Tensor 
from torch import float as floatt
from torch import long as longt
from torch.optim import Optimizer

QA = Tuple[str, str]

MayTensor = Maybe[Tensor]


@dataclass 
class Box:
    # coordinates in (x,y) cv2-like frame
    x: int 
    y: int
    w: int 
    h: int 


@dataclass
class Object:
    label: str
    category: str
    box: Box 


@dataclass
class ObjectCrop(Object):
    image: array 


@dataclass
class Scene:
    environment: str
    image_id: str
    rgb: array
    depth: array
    objects: Sequence[Object]

    @property
    def labels(self):
        return [o.label for o in self.objects]

    @property
    def boxes(self):
        return [o.box for o in self.objects]

    def get_crops(self):
        return [self.rgb[o.box.y : o.box.y+o.box.h, o.box.x : o.box.x+o.box.w] for o in self.objects]
    

@dataclass
class AnnotatedObject(Object):
    color: List[str]
    # shape: List[str]
    # material: List[str]
    # general: List[str]
    special: List[str]


@dataclass
class SceneGraph:
    nodes: Sequence[AnnotatedObject]
    edges: array


@dataclass
class AnnotatedScene(Scene):
    query: List[List[str]]
    truth: List[List[int]]
    # qas: List[QA]
    # infs: List[QA]
    
