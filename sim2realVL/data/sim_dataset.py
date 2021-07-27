from ..types import *
from ..utils.image_proc import * 

import pandas as pd
import os
import cv2
from functools import lru_cache

CATEGORY_MAP = {
    'mug'       :   'coffee mug',
    'bowl'      :   'bowl',
    'can'       :   'soda can',
    'cereal'    :   'cereal box',
    'cap'       :   'cap',
    'flashlight':   'flashlight'
}


class SimScenesDataset:
    def __init__(self, images_path: str, csv_path: str):
        self.root = images_path
        self.table = pd.read_table(csv_path)
        self.image_ids = self.table['image_ids'].tolist()
        self.labels = [row.split(',') for row in self.table['labels'].tolist()]
        self.boxes =  [row.split('),') for row in self.table['boxes'].tolist()]
        self.boxes = [[b.split(',') for b in row] for row in self.boxes]
        self.boxes = [[[x.strip("()") for x in b] for b in c] for c in self.boxes]
        self.boxes = [[Box(*list(map(int, b))) for b in row] for row in self.boxes]
        self.depths = [list(map(float, d.split(','))) for d in self.table['depths'].tolist()]
        self.categories = [[CATEGORY_MAP[l.split('_')[0]] for l in labs] for labs in self.labels]
        self.objects = [[Object(l, c, b) for l, c, b in zip(ls, cs, bs)] 
                        for ls, cs, bs in zip(self.labels, self.categories, self.boxes)]

    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, str(self.image_ids[n]) + '.jpg'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n: int) -> Scene:
        return Scene(environment="sim", 
                     image_id=self.image_ids[n], 
                     rgb=self.get_image(n), 
                     depth=self.depths[n],
                     objects=self.objects[n])

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = scene.rgb.copy()
        for obj in scene.objects:
            img = cv2.putText(img, obj.label, (obj.box.x, obj.box.y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (obj.box.x, obj.box.y), (obj.box.x+obj.box.w, obj.box.y+obj.box.h), (0,0,0xff), 2)
        show(img, str(self.image_ids[n]) + '.jpg')

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n, str(self.image_ids[n]) + '.jpg')


class SimScenesVGDataset(SimScenesDataset):
    def __init__(self, images_path, csv_path):
        super().__init__(images_path, csv_path)
        self.querries = sum([row.split(',') for row in self.table['querries'].tolist()], [])
        self.image_ids_flat = sum([[iid] * len(l) for iid, l in zip(self.image_ids, self.labels)], [])
        self.depths_flat = sum([[d] * len(d) for d in self.depths], [])
        self.objects_flat = sum([[o] * len(o) for o in self.objects], [])
        self.truths = sum([np.eye(len(l), dtype=int).tolist() for l in self.labels], [])

    @lru_cache(maxsize=None)
    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, str(self.image_ids_flat[n]) + '.jpg'))

    def __len__(self):
        return len(self.querries)

    def __getitem__(self, n: int) -> AnnotatedScene:
        return AnnotatedScene(environment="sim",
                              image_id=self.image_ids_flat[n],
                              rgb=self.get_image(n),
                              depth=self.depths_flat[n],
                              objects=self.objects_flat[n],
                              query=self.querries[n],
                              truth=self.truths[n]
                              )
    def show(self, n: int):
        scene = self.__getitem__(n)
        print(self.querries[n])
        img = scene.rgb.copy()
        for obj, truth in zip(scene.objects, scene.truth):
            if not truth:
                continue
            img = cv2.putText(img, obj.label, (obj.box.x, obj.box.y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (obj.box.x, obj.box.y), (obj.box.x+obj.box.w, obj.box.y+obj.box.h), (0,0,0xff), 2)
        show(img, str(self.image_ids_flat[n]) + '.jpg')


def get_sim_rgbd_scenes():
    return SimScenesDataset("datasets/SIM/rgbd-scenes/Images", "datasets/SIM/rgbd-scenes/data.csv")


def get_sim_rgbd_scenes_vg():
    return SimScenesVGDataset("datasets/SIM/rgbd-scenes/Images", "datasets/SIM/rgbd-scenes/data_vg.csv")