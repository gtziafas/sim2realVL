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


class SimScenesOldDataset:
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
            self.show(n)


class SimScenesDataset:
    def __init__(self, images_path: str, csv_path: str):
        self.root = images_path
        self.table = pd.read_table(csv_path)
        self.image_ids = self.table['image_id'].tolist()
        self.labels = [row.split(',') for row in self.table['label'].tolist()]
        self.pos_2d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['2D_position'].tolist()]
        self.pos_2d = [[(p[i], p[i+1]) for i in range(0, len(p)-1, 2)] for p in self.pos_2d]
        self.centers = [[int(x.strip("()")) for x in c.split(',')] for c in self.table['RGB_center_of_mass'].tolist()]
        self.centers = [[(c[i], c[i+1]) for i in range(0, len(c)-1, 2)] for c in self.centers]
        self.boxes = [[int(x.strip("()")) for x in b.split(',')] for b in self.table['RGB_bounding_box'].tolist()]
        self.boxes = [[Box(*b[i:i+4]) for i in range(0, len(b)-1, 4)] for b in self.boxes]
        self.rects = [[int(x.strip("()")) for x in r.split(',')] for r in self.table['RGB_rotated_box'].tolist()]
        self.rects = [[Rectangle(*r[i:i+8]) for i in range(0, len(r)-1, 8)] for r in self.rects]
        self.categories = [[CATEGORY_MAP[l.split('_')[0]] for l in labs] for labs in self.labels]
        self.objects = [[ObjectSim(l, cat, b, r, c, p) for l, cat, p, c, b, r in zip(ls, cats, ps, cs, bs, rs)] 
                        for ls, cats, ps, cs, bs, rs in zip(self.labels, self.categories, self.pos_2d,
                        self.centers, self.boxes, self.rects)]

    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, str(self.image_ids[n]) + '.png'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n: int) -> Scene:
        return Scene(environment="sim", 
                     image_id=self.image_ids[n], 
                     objects=self.objects[n])

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = self.get_image(n).copy()
        for obj in scene.objects:
            x, y, w, h = obj.bounding_box
            img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0xff), 2)
        show(img, str(self.image_ids[n]))

    def show_id(self, id: int):
        self.show(self.image_ids.index(id))

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n)

    def massage(self, from_, to_):
        drop = []
        for i in range(from_, to_):
            scene = self.__getitem__(i)
            img = self.get_image(i).copy()
            for obj in scene.objects:
                x, y, w, h = obj.bounding_box.x, obj.bounding_box.y, obj.bounding_box.w, obj.bounding_box.h
                img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0xff), 2)    
        
            cv2.imshow(str(self.image_ids[i]), img)
            while True:
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    drop.append(self.image_ids[i])
                    break
            cv2.destroyWindow(str(self.image_ids[i]))
        return drop


class SimScenesVGOldDataset(SimScenesDataset):
    def __init__(self, images_path, csv_path):
        super().__init__(images_path, csv_path)
        self.querries = sum([row.split(',') for row in self.table['querries'].tolist()], [])
        self.image_ids_flat = sum([[iid] * len(l) for iid, l in zip(self.image_ids, self.labels)], [])
        self.depths_flat = sum([[d] * len(d) for d in self.depths], [])
        self.objects_flat = sum([[o] * len(o) for o in self.objects], [])
        self.truths = sum([np.eye(len(l), dtype=int).tolist() for l in self.labels], [])

    #@lru_cache(maxsize=None)
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


def get_sim_rgbd_scenes_old():
    return SimScenesOldDataset("datasets/SIM/rgbd-scenes_old/Images", "datasets/SIM/rgbd-scenes_old/data_big.csv")


def get_sim_rgbd_scenes_vg_old():
    return SimScenesVGOldDataset("datasets/SIM/rgbd-scenes_old/Images", "datasets/SIM/rgbd-scenes_old/data_vg_big.csv")


def get_sim_rgbd_scenes():
    return SimScenesDataset("datasets/SIM/rgbd-scenes/Images", "datasets/SIM/rgbd-scenes/data.csv")


def get_sim_rgbd_scenes_vg():
    return SimScenesVGOldDataset("datasets/SIM/rgbd-scenes/Images", "datasets/SIM/rgbd-scenes/data_vg.csv")
