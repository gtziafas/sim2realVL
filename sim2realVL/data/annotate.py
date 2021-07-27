from ..types import * 
from ..utils.image_proc import *
from ..data.graphs import *
from ..data.rgbd_scenes import RGBDScenesDataset

from random import random, choice, seed
from tqdm import tqdm
import pandas as pd
from collections import Counter

seed(1312)


def spatial_replace(idx: str, graph: SceneGraph, chance: float):
    def inner_chance(token_idx: int) -> str:
        if chance < 0.5:
            return  graph.nodes[token_idx].category
        elif chance < 0.8:
            return  graph.nodes[token_idx].color + ' ' + graph.nodes[token_idx].category
        else:
            return choice(graph.nodes[token_idx].special) 
            
    # identify closest object and make sure its not the ambiguous one
    dists = graph.edges[idx, :, -1]
    cat, cats = graph.nodes[idx].category, [o.category for o in graph.nodes]
    closest = informed_argmin(dists, cat, cats)
    x_rel = graph.edges[idx, closest, 3]
    z_rel = graph.edges[idx, closest, 4]
    x_dist = box_center(graph.nodes[idx].box)[0] - box_center(graph.nodes[closest].box)[0] 
    y_dist = box_center(graph.nodes[idx].box)[1] - box_center(graph.nodes[closest].box)[1]

    # use it if the distance itself is not very big (temp thresh)
    if dists[closest] <= 75:
        if z_rel == 0 or abs(x_dist) > 120:
            return f'The {cat} next to the {inner_chance(closest)}'
        elif z_rel == 1:
            return f'The {cat} behind the {inner_chance(closest)}'
        else:
            return f'The {cat} in front of the {inner_chance(closest)}'

    elif dists[closest] <= 200:
        return f'The {cat} close to the {inner_chance(closest)}'
    
    # else use left/right
    else:
        if x_rel == 1:
            return f'The {cat} right from the {inner_chance(closest)}'
        else:
            return f'The {cat} left from the {inner_chance(closest)}'
      

def visual_replace(idx: int, graph: SceneGraph):
    obj = graph.nodes[idx]

    chance = random()
    if chance < 0.5:
        # keep category
        return obj.category
    
    elif chance < 0.75:
        # replace with COLOR + category
        return obj.color + ' ' + obj.category
    
    else:
        # replace with special tag from catalogue
        return choice(obj.special)
       


def randomly_replace(idx: int, graph: SceneGraph):
    obj = graph.nodes[idx]

    if len(graph.nodes) > 1:
        chance = random()
        
        # FOR REAL IT WAS 0.75!!!!
        if chance < 0.5:
            return visual_replace(idx, graph)        
        else:
            # replace with spatial relationship
            return spatial_replace(idx, graph, chance=random())

    else:
        # if only one object dont do spatial
        return visual_replace(idx, graph)
                

def generate_refex(scenes: List[Scene]) -> List[List[str]]:
    
    def _generate_refex(scene: Scene) -> List[str]:

        graph = extract_scene_graph(scene)
        labels = [o.label for o in graph.nodes]
        cats = [o.category for o in graph.nodes]
        colors = [o.color for o in graph.nodes]
        tags = [o.special for o in graph.nodes]
        relations = graph.edges
        _refex = []

        # if all objects unique, give label
        if len(set(cats)) == len(cats):
            _refex.extend(cats)

        else:
            # split them into groups of unique categories and count group sizes
            _cats_counter = Counter(cats)
            # _multi_cat, _multi_color = set({}), set({})
            
            for idx, (cat, color) in enumerate(zip(cats, colors)):
                cat_count = _cats_counter[cat]
                color_count = len(set([colors[i] for i, c in enumerate(cats) if c == cat]))

                # give a sample for unique items (works also for multi-labeled cases)
                if cat_count == 1:
                    _refex.append(cat)

                else:
                    # keep track to add manually 
                    # _multi_cat.add(cat))

                    # give color + label sample (works also for multi-labeled cases)
                    if color_count == cat_count:
                        chance = random()
                        if chance < 0.5:
                            _refex.append(color + ' ' + cat)
                        else:
                            _refex.append(choice(graph.nodes[idx].special))
                            #_refex.append(color + ' ' + cat)

                    # else we have to use spatial relationship
                    else:
                        # keep track to add manually
                        # _multi_color.add(color + ' ' + cat))
                        _refex.append(spatial_replace(idx, graph, chance=0.))

            # if _multi_cat:
            #     _refex.extend(list(_multi_cat))

            # if _multi_color:
            #     _refex.extend(list(_multi_color))    
        return _refex

    refex = []
    for scene in tqdm(scenes):
        refex.append(_generate_refex(scene))
    return massage_refex(refex, scenes)


def massage_refex(refex: List[List[str]], scenes: List[Scene]):
    previous = []
    for idx, (_refex, scene) in enumerate(zip(refex, scenes)):
        graph = extract_scene_graph(scene)

        # FOR REAL
        # # if same as previous, randomly add some content to some captions
        # if _refex != previous:
        #     previous = _refex
        #     continue

        # else:
            # identify objects that are captioned by their label
            # obj_ids =  [i for i, obj in enumerate(graph.nodes) if obj.category == _refex[i]]
            # _refex = [randomly_replace(i, graph) if i in obj_ids else c for i, c in enumerate(_refex)]
            # refex[idx] = _refex
    
        # FOR SIM
        obj_ids =  [i for i, obj in enumerate(graph.nodes) if obj.category == _refex[i]]
        _refex = [randomly_replace(i, graph) if i in obj_ids else c for i, c in enumerate(_refex)]
        refex[idx] = _refex        

    return refex
                        

def informed_argmin(arr: array, here: str, info: List[str]) -> int:
    closest = arr.argmin()
    if info[closest] == here:
        arr[closest] = 1e10
        return informed_argmin(arr, here, info)
    else:
        return closest


def _make_refex_dataset(ds: List[Scene], refex: List[List[str]], save_path: str):
    header = '\t'.join(['no','scene','subfolder','image_id','objects','labels','boxes','query','truth'])
    with open(save_path, 'w+') as f:
        f.write(header)
        f.write('\n')

    for i, (scene, querries) in enumerate(zip(ds, refex)):
        # order from left to right and bottom to top
        zipped = sorted(zip(scene.objects, querries), key=lambda t: (t[0].box.x, t[0].box.y))
        objects, querries = zip(*zipped)

        subfolder = scene.environment
        img_id = scene.image_id
        cats = ",".join([o.category for o in objects])
        labels = ",".join([o.label for o in objects])
        boxes = ",".join([str((o.box.x, o.box.y, o.box.w, o.box.h)) for o in objects])
        truths = [str(tuple(x.astype(int))).replace(' ','') for x in list(np.eye(len(objects)))]
        
        for j, (q, t) in enumerate(zip(querries, truths)):
            strs = '\t'.join([str(i+j+1), 
                              '_'.join(subfolder.split('_')[:-1]),
                              subfolder,
                              str(img_id),
                              cats,
                              labels,
                              boxes,
                              q, t]
                            )
            with open(save_path, 'a+') as f:
                f.write(strs)
                f.write('\n')


def make_refex_dataset(ds: List[Scene], save_path: str):
    _refex = generate_refex(ds)
    _make_refex_dataset(ds, _refex, save_path)


def make_refex_dataset_sim(csv_path: str, refex: List[List[str]], save_path: str):
    table = pd.read_table(csv_path)
    iids = table['image_ids'].tolist()
    labels = table['labels'].tolist()
    boxes = table['boxes'].tolist()
    depths = table['depths'].tolist()
    refex = [','.join(r) for r in refex]
    WRITE = ['\t'.join([str(i), l, b, d, r]) for i, l, b, d, r in zip(iids, labels, boxes, depths, refex)]
    with open(save_path, "w") as f:
        f.write('\t'.join(["image_ids", "labels", "boxes", "depths", "querries"]))
        f.write('\n')
        f.write('\n'.join(WRITE))