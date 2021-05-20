from typing import List, Tuple, Dict, Any, Callable, Set, TypeVar, Sequence
from typing import Optional as Maybe
from typing import Mapping as Map 
from dataclasses import dataclass 
from abc import ABC 

from numpy import array


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
    image: array 
    box: Box


@dataclass
class Scene:
    environment: str
    rgb: array
    depth: array
    objects: Sequence[Object]


@dataclass
class AnnotatedObject(Object):
    color: List[str]
    shape: List[str]
    material: List[str]
    general: List[str]
    special: List[str]

    def __init__(self, 
                 color = None, 
                 shape = None, 
                 material = None, 
                 general = None, 
                 special = None):
        color = color if color is not None else []
        shape = shape if shape is not None else []
        material = material if material is not None else []
        general = general if general is not None else []
        special = general if general is not None else []
