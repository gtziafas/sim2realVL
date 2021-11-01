from ..types import *
from ..utils.image_proc import * 

import os
import cv2
import subprocess
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from scipy.io import loadmat
import torch

ROOT = 'datasets/rgbd-scenes'
BOXES_TSV = 'rgbd-scenes_boxes.tsv'
FEATURES_BINARY = 'checkpoints/resnet18_features_tuned1.p'
CATALOGUE_TSV ='datasets/rgbd-scenes/rgbd-scenes_catalogue.csv'
VG_TSV = 'rgbd-scenes_vg_v2.tsv' 


# split image folders to rgb/ and depth/ sub-directiories for convinience
def split_images_to_rgb_depth(datapath: str):
    scenes = os.listdir(datapath)
    for scene in tqdm(scenes):
        scene_path = os.path.join(datapath, scene)
        folds = [f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        for fold in folds:
            path = os.path.join(scene_path, fold)
            imgs = os.listdir(path)

            # align each rgb and depth image
            rgb = sorted([i for i in imgs if 'depth' not in i])
            depth = sorted([i for i in imgs if 'depth' in i], key = lambda s: s.split('_depth'))
            
            # create sub-dirs for rgb and depth seperately
            if not os.path.isdir(os.path.join(path, 'rgb')):
                os.mkdir(os.path.join(path, 'rgb'))
            if not os.path.isdir(os.path.join(path, 'depth')):
                os.mkdir(os.path.join(path, 'depth'))

            # split them by bash renaming
            for r, d in list(zip(rgb, depth)):
                subprocess.call(['mv', os.path.join(path, r), os.path.join(path, 'rgb', r)])
                subprocess.call(['mv', os.path.join(path, d), os.path.join(path, 'depth', d)])


# convert .mat format files to python lists
def parse_mat_format(bboxes: np.ndarray):
    categories, instances, boxes = [], [], [] 
    for box in bboxes:
        try:
            categories.append(box['category'][0].tolist())
            instances.append(box['instance'][0].tolist())
            tops = box['top'][0].tolist()
            bottoms = box['bottom'][0].tolist()
            lefts = box['left'][0].tolist()
            rights = box['right'][0].tolist()
            boxes.append(list(zip(tops, bottoms, lefts, rights)))
        except IndexError:
            categories.append([])
            instances.append([])
            boxes.append([])
        
    categories = [[s[0] for s in cat] for cat in categories]
    instances = [[s[0][0] for s in inst] for inst in instances]
    boxes = [[(t[0][0],b[0][0],l[0][0],r[0][0]) for t,b,l,r in box] for box in boxes]
    labels = [[str(c)+'_'+str(j) for c,j in zip(cat, inst)] for cat, inst in zip(categories, instances)]

    return categories, instances, labels, boxes


# create a single csv with all label and bounding box annotations
def create_csv_with_bboxes(root: str = ROOT, csv_name: str = BOXES_TSV):
    pass # jesus


class FromTableDataset:
    def __init__(self, root: str = ROOT, table_file: str = BOXES_TSV):
        self.name = table_file.split('_')[0]
        self.root = root
        self.table = pd.read_table(os.path.join(root, table_file))
        self.parse_table()
        self.unique_scenes = sorted(list(set(self.rgb_paths)))

    def parse_table(self):
        self.scenes = self.table['scene'].tolist()
        self.environments = self.table['subfolder'].tolist()
        self.image_ids = self.table['image_id'].tolist()
        self.categories = [' '.join(cat.split('_')) for cat in self.table['object'].tolist()]
        self.labels = self.table['label'].tolist()
        self.boxes = [Box(*tuple(map(int, b.strip('()').split(',')))) for b in self.table['box']]
        self.rgb_paths = [os.path.join(self.root, 
                                        scene,
                                        env,
                                        'rgb',
                                        '_'.join([env, str(iid)]) + '.png')
                            for scene, env, iid in zip(self.scenes, self.environments, self.image_ids)]

    @lru_cache(maxsize=None)
    def get_image(self, n: int) -> array:
        return cv2.imread(self.unique_scenes[n])

    @lru_cache(maxsize=None)
    def get_depth(self, n: int) -> array:
        scene_path = self.unique_scenes[n].split('/')
        token = scene_path[-1].split('.png')[0] + '_depth.png'
        depth_path = '/'.join(scene_path[:-2] + ['depth', token])
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        return (depth / depth.max() * 0xff).astype('uint8')


class RGBDObjectsDataset(FromTableDataset):

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, n: int) -> Object:
        unique_idx = self.unique_scenes.index(self.rgb_paths[n])
        img = self.get_image(unique_idx)
        return ObjectCrop(label = self.labels[n], 
                      category = self.categories[n],
                      image = crop_box(img, self.boxes[n]),
                      box = self.boxes[n])


class RGBDScenesDataset(FromTableDataset):

    def __init__(self):
        super().__init__()
        self.update_properties()

    def __len__(self) -> int:
        return len(self.unique_scenes)

    def __getitem__(self, n: int) -> Scene:
        # in scenes version we do not need to save each box
        objects = [Object(label=l, category=c, box=b) for l, c, b in zip(self.labels[n], self.categories[n], self.boxes[n])]
        iid = self.unique_scenes[n].split('_')[-1].split('.png')[0]
        return Scene(environment=self.environments[n], image_id=iid, objects=objects)

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = self.get_image(n)
        for obj in scene.objects:
            img = cv2.putText(img, obj.label, (obj.box.x, obj.box.y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (obj.box.x, obj.box.y), (obj.box.x+obj.box.w, obj.box.y+obj.box.h), (0,0,0xff), 2)
        show(img)

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n)

    def update_properties(self):
        idces = [[i for i, p in enumerate(self.rgb_paths) if p == scene] for scene in self.unique_scenes]
        self.labels = [[self.labels[i] for i in ids] for ids in idces]
        self.boxes = [[self.boxes[i] for i in ids] for ids in idces]
        self.environments = [[self.environments[i] for i in ids] for ids in idces]
        assert [len(set(e)) for e in self.environments] == [1] * len(self.environments)
        self.environments = [e[0] for e in self.environments]
        self.categories = [[self.categories[i] for i in ids] for ids in idces]
    

class FromBinaryFeatures:
    def __init__(self, binary_path: str = FEATURES_BINARY):
        self.features = torch.load(binary_path)


class RGBDObjectsFeatures(FromBinaryFeatures):

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, n: int) -> Tensor:
        return self.features[n]


class RGBDScenesFeatures(FromTableDataset, FromBinaryFeatures):
    def __init__(self):
        FromTableDataset.__init__(self)
        FromBinaryFeatures.__init__(self)

    def __len__(self) -> int:
        return len(self.unique_scenes)

    def __getitem__(self, n: int) -> Tensor:
        path = self.unique_scenes[n]
        idces = [i for i, p in enumerate(self.rgb_paths) if p == path]
        return torch.stack([self.features[n] for n in idces])


class RGBDScenesVG:
    def __init__(self, root: str = ROOT, table_file: str = VG_TSV):
        self.name = table_file.split('_')[0]
        self.root = root
        self.table = pd.read_table(os.path.join(root, table_file))

        # parse table
        self.scenes = self.table['scene'].tolist()
        self.labels = [row.split(',') for row in self.table['labels'].tolist()]
        self.boxes = [[Box(*tuple(map(int, b.strip('()').split(',')))) for b in row.split(',(')] 
                    for row in self.table['boxes'].tolist()]
        self.categories = [row.split(',') for row in self.table['objects'].tolist()]
        self.refex = self.table['query'].tolist()
        self.truths = [[int(t) for t in row.replace(' ','').strip('()').split(',') if t != ''] for row in self.table['truth'].tolist()]
        self.environments = self.table['subfolder'].tolist()
        self.image_ids = self.table['image_id'].tolist()
        self.rgb_paths = [os.path.join(self.root, scene, env, 'rgb', '_'.join([env, str(iid)]) + '.png')
                            for scene, env, iid in zip(self.scenes, self.environments, self.image_ids)]

        self.unique_scenes = sorted(list(set(self.rgb_paths)))

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, n: int) -> AnnotatedScene:
        unique_idx = self.unique_scenes.index(self.rgb_paths[n])
        rgb, depth = self.get_image(unique_idx), self.get_depth(unique_idx)

        objects = [Object(label=l, category=c, box=b) for l, c, b in zip(self.labels[n], self.categories[n], self.boxes[n])]
        return AnnotatedScene(environment=self.environments[n], 
                              image_id = self.image_ids[n],
                              rgb = rgb, 
                              depth = depth, 
                              objects = objects, 
                              query = self.refex[n], 
                              truth = self.truths[n])
    
    @lru_cache(maxsize=None)
    def get_image(self, n: int) -> array:
        return cv2.imread(self.unique_scenes[n])
    
    @lru_cache(maxsize=None)
    def get_depth(self, n: int) -> array:
        scene_path = self.unique_scenes[n].split('/')
        token = scene_path[-1].split('.png')[0] + '_depth.png'
        depth_path = '/'.join(scene_path[:-2] + ['depth', token])
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        return (depth / depth.max() * 0xff).astype('uint8')

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = scene.rgb.copy()
        obj = scene.objects[array(scene.truth).argmax()]
        print(scene.query)
        img = cv2.putText(img, obj.label, (obj.box.x, obj.box.y), fontFace=0, fontScale=1, color=(0,0,0xff))
        img = cv2.rectangle(img, (obj.box.x, obj.box.y), (obj.box.x+obj.box.w, obj.box.y+obj.box.h), (0,0,0xff), 2)
        show(img)

    def inspect(self):
        idces = random.sample(list(range(self.__len__())), self.__len__())
        for idx in idces:
            self.show(idx)

def get_object_catalogue(table_path: str = CATALOGUE_TSV):
    return pd.read_table(table_path, sep=',')
 