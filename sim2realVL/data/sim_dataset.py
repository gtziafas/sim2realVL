from ..types import *
from ..utils.image_proc import * 

import pandas as pd
import subprocess
import os
import cv2
from random import sample
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
        self.csv_path = csv_path
        self.table = pd.read_table(csv_path)
        self.image_ids = self.table['image_id'].tolist()
        self.labels = [row.split(',') for row in self.table['label'].tolist()]
        self.contours =  [[eval(x.strip("()")) for x in row.split("),")] for row in self.table["RGB_contour"].tolist()]
        self.contours = [[np.int0([[[c[i], c[i+1]]] for i in range(0, len(c)-1, 2)]) for c in row] for row in self.contours]
        self.pos_2d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['2D_position'].tolist()]
        self.pos_2d = [[(p[i], p[i+1]) for i in range(0, len(p)-1, 2)] for p in self.pos_2d]
        moments = [[cv2.moments(c) for c in row] for row in self.contours]
        self.centers = [[(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in row] for row in moments]
        self.boxes = [[Box(*cv2.boundingRect(c)) for c in row] for row in self.contours]
        self.rects = [[Rectangle(*sum(cv2.boxPoints(cv2.minAreaRect(c)).tolist(), [])) for c in row] for row in self.contours]          
        # self.centers = [[int(x.strip("()")) for x in c.split(',')] for c in self.table['RGB_center_of_mass'].tolist()]
        # self.centers = [[(c[i], c[i+1]) for i in range(0, len(c)-1, 2)] for c in self.centers]
        # self.boxes = [[int(x.strip("()")) for x in b.split(',')] for b in self.table['RGB_bounding_box'].tolist()]
        # self.boxes = [[Box(*b[i:i+4]) for i in range(0, len(b)-1, 4)] for b in self.boxes]
        # self.rects = [[int(x.strip("()")) for x in r.split(',')] for r in self.table['RGB_rotated_box'].tolist()]
        # self.rects = [[Rectangle(*r[i:i+8]) for i in range(0, len(r)-1, 8)] for r in self.rects]
        self.categories = [[CATEGORY_MAP[l.split('_')[0]] for l in labs] for labs in self.labels]
        self.objects = [[ObjectSim(l, cat, co, b, r, c, p) for l, cat, co, p, c, b, r in zip(ls, cats, cos, ps, cs, bs, rs)] 
                        for ls, cats, cos, ps, cs, bs, rs in zip(self.labels, self.categories, self.contours,
                        self.pos_2d, self.centers, self.boxes, self.rects)]  

    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, str(self.image_ids[n]) + '.png'))

    def get_image_from_id(self, image_id: int) -> array:
        return cv2.imread(os.path.join(self.root, str(image_id) + '.png'))

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
            x, y, w, h = obj.box.x, obj.box.y, obj.box.w, obj.box.h
            rect = obj.rectangle
            rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff, 0, 0), 2)
            img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
            img = cv2.drawContours(img, [obj.contour], 0, (0,0xff, 0), 1)
        show(img, str(self.image_ids[n]) + ".png")

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
                x, y, w, h = obj.box.x, obj.box.y, obj.box.w, obj.box.h
                rect = obj.rectangle
                rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
                img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff,0,0  ), 2)    
                img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
                img = cv2.drawContours(img, [obj.contour], 0, (0,0xff,0), 2)
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


class SimScenesAnnotatedDataset():
    def __init__(self, root: str, csv_path: str, annot_path: str):
        self.scenes = SimScenesDataset(root, csv_path)
        self.root = root
        self.table = pd.read_table(annot_path)
        self.queries = [[q.strip("'") for q in qs.strip(']["').split(', ')] for qs in self.table["queries"].tolist()] 
        self.truths = [[eval(t.strip("'")) for t in ts.strip(']["').split(', ')] for ts in self.table["truths"].tolist()] 
        self.truths = [[list(t) if type(t) == tuple else t for t in ts] for ts in self.truths]
        self.scenes = sum([[scene] * len(qs) for scene, qs in zip(self.scenes, self.queries)], [])
        self.image_ids = [scene.image_id for scene in self.scenes]
        self.queries = sum(self.queries, [])
        self.truths = sum(self.truths, [])

    def get_image(self, n: int) -> array:
        return self.get_image_from_id(self.scenes[n].image_id)

    def get_image_from_id(self, image_id: int) -> array:
        return cv2.imread(os.path.join(self.root, str(image_id) + '.png'))

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, n: int) -> AnnotatedScene:
        scene = self.scenes[n]
        return AnnotatedScene(scene.environment, scene.image_id, scene.objects,
                              query=self.queries[n], truth=self.truths[n])

    def show(self, n: int):
        scene = self.__getitem__(n)
        print(scene.query)
        img = self.get_image_from_id(scene.image_id)
        truth = [scene.truth] if type(scene.truth) == int else list(scene.truth)
        boxes = [o.box for i, o in enumerate(scene.objects) if i in truth]
        for b in boxes:
            img = cv2.rectangle(img, (b.x, b.y), (b.x+b.w, b.y+b.h), (0xff,0,0), 2)
        show(img)

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n)

    def filter(self, idces: List[int]):
        self.queries = [q for i, q in enumerate(self.queries) if i in idces]
        self.truths = [t for i, t in enumerate(self.truths) if i in idces]
        self.scenes = [s for i, s in enumerate(self.scenes) if i in idces]
        self.image_ids = [iid for i, iid in enumerate(self.image_ids) if i in idces]


class SimScenesVGOldDataset(SimScenesOldDataset):
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
    return SimScenesDataset("/home/p300488/dual_arm_ws/DATASET/Images", "/home/p300488/dual_arm_ws/DATASET/data.tsv")


def get_sim_rgbd_objects(size: int = -1):
    ds = get_sim_rgbd_scenes()
    size = size if size > -1 else len(ds)
    keep_idces = sample(range(len(ds)), size)
    crops, labels = [], []
    for i, scene in enumerate(ds):
        if i  not in keep_idces:
            continue
        rgb = ds.get_image(i)
        crops.extend([crop_contour(mask, o.contour) for o in scene.objects])
        labels.extend([o.label for o in scene.objects])
    return list(zip(crops, labels))


def get_sim_rgbd_scenes_vg():
    return SimScenesVGOldDataset("datasets/SIM/rgbd-scenes/Images", "datasets/SIM/rgbd-scenes/data_vg.csv")


def get_sim_rgbd_scenes_annotated(split: str = "all"):
    ds = SimScenesAnnotatedDataset("/home/p300488/dual_arm_ws/DATASET/Images", 
                        "/home/p300488/dual_arm_ws/DATASET/data.tsv",
                        "datasets/SIM/rgbd_scenes/data_annotated.csv")
    if split == "all":
        return ds

    elif split == "spatial":
        keywords = ['left', 'right', 'behind', 'front', 'next']
        idces = []
        for i, scene in enumerate(ds):
            tokens = scene.query.split()
            if len(tokens) > 3 or (len(tokens) == 3 and tokens[1] in keywords):
                idces.append(i)
        ds.filter(idces)
        return ds
    
    elif split == "category":
        cats = set(sum([s.categories for s in ds], []))
        idces = [i for i, s in enumerate(ds) if s.query in cats]
        ds.filter(idces)
        return ds 

    elif split == "color":
        colors = ['blue', 'green', 'white', 'black', 'red', 'purple', 'brown', 'orange', 'yellow']
        idces = [i for i, s in enumerate(ds) if s.query.split()[0] in colors]
        ds.filter(idces)
        return ds


def remove_samples_from_dataset(ds: SimScenesDataset, sample_ids: List[int]):
    for sid in sample_ids:
        subprocess.call(['rm', os.path.join(ds.root, str(sid) + ".png")])
    csv_name = ds.csv_path.split('.csv')[0] + "_filtered.csv"
    with open(csv_name, "w+") as f:
        f.write("\t".join(["image_id", "label", "2D_position", "RGB_center_of_mass", "RGB_bounding_box", "RGB_rotated_box"]))
        f.write("\n")
        _write = ["\t".join(list(map(str, row))) for row in ds.table.values.tolist() if row[0] not in sample_ids]
        f.write("\n".join(_write))