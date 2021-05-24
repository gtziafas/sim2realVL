from ..types import *
from ..utils.image_proc import * 
from ..data.rgbd_scenes import get_object_catalogue

SYMMETRIES = {
    'left from'     : 'right from',
    'above'         : 'below',
    'behind'        : 'in front of',
    'next to'       : 'next to',
    'alongisde'     : 'alongisde',
    'far from'      : 'far from',
    'bigger than'   : 'smaller than',

}

CATALOGUE = get_object_catalogue()
COLOR_MAP = {k: v.split(',')[0] for k, v in zip(CATALOGUE['label'], CATALOGUE['color'])}
SPECIAL_MAP = {k: [s.strip() for s in v.split(',')] for k, v in zip(CATALOGUE['label'], CATALOGUE['special'])}

RelationVector = Tuple[float, ...] # (6,)


def compare_size(obj1: AnnotatedObject, obj2: AnnotatedObject) -> int:
    size1 = obj1.box.h * obj1.box.w
    size2 = obj2.box.h * obj2.box.w
    ratio = size1 / size2
    return 0 if 0.95 <= ratio <= 1.05 else 1 if ratio > 1.05 else -1


def compare_height(obj1: AnnotatedObject, obj2: AnnotatedObject) -> int:
    h1 = obj1.box.h; h2 = obj2.box.h
    ratio = h1 / h2
    return 0 if 0.95 <= ratio <= 1.05 else 1 if ratio > 1.05 else -1


def compare_width(obj1: AnnotatedObject, obj2: AnnotatedObject) -> List[str]:
    w1 = obj1.box.w; w2 = obj2.box.w
    ratio = w1 / w2
    return 0 if 0.95 <= ratio <= 1.05 else 1 if ratio > 1.05 else -1


def compare_x(obj1: AnnotatedObject, obj2: AnnotatedObject, max_width: int = 480) -> int:
    c1 = (obj1.box.x + obj1.box.w // 2, obj1.box.y + obj1.box.h // 2)
    c2 = (obj2.box.x + obj2.box.w // 2, obj2.box.y + obj2.box.h // 2)
    return 0 if abs(c1[0] - c2[0]) <= 10 else 1 if c1[0] > c2[0] else -1 
    #dist_x = (c1[0] - c2[0]) / max_width
    #return 0 if -0.05 <= dist_x <= 0.05 else 1 if dist_x > 0.05 else -1


def compare_z(obj1: AnnotatedObject, obj2: AnnotatedObject, depth: array) -> int:
    c1 = (obj1.box.x + obj1.box.w // 2, obj1.box.y + obj1.box.h // 2)
    c2 = (obj2.box.x + obj2.box.w // 2, obj2.box.y + obj2.box.h // 2)
    # take the average over a 10x10 window with removed noise
    dist_z = depth_eval(c1, depth) - depth_eval(c2, depth)
    return 0 if -10 < dist_z < 10 else 1 if dist_z > 0 else -1


def compare_distance(obj1: AnnotatedObject, obj2: AnnotatedObject, depth: array) -> float:
    c1, c2 = box_center(obj1.box), box_center(obj2.box)
    d1, d2 = depth_eval(c1, depth), depth_eval(c2, depth)
    dist =  np.linalg.norm(array([c1[0] - c2[0], c1[1] - c2[1], d1 - d2])) # Euclidean
    # dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(d1 - d2) # Manhattan
    return dist
    #return 0 if abs(dist) < dist_thresh else 1 


def extract_scene_graph(scene: Scene) -> SceneGraph:
    colors = [COLOR_MAP[l] for l in scene.labels]
    tags = [SPECIAL_MAP[l] for l in scene.labels]
    
    annot_objects = [AnnotatedObject(label=o.label, category=o.category, box=o.box, color=colors[i], special=tags[i]) 
                     for i, o in enumerate(scene.objects)]
    relations = np.empty((len(scene.objects), len(scene.objects), 6), dtype=float) 

    sizes = [o.box.h * o.box.w for o in annot_objects]
    hs, ws = zip(*[(o.box.h, o.box.w) for o in annot_objects])
    xs = [o.box.x + o.box.w // 2 for o in annot_objects]
    zs = [depth_eval((o.box.x + o.box.w // 2, o.box.y + o.box.h // 2), scene.depth) for o in annot_objects]
    for i, o in enumerate(annot_objects):

        # add main diagonal relation element
        relations[i, i] = array([-1 if sizes[i] == min(sizes) else 1 if sizes[i] == max(sizes) else 0,
                                 -1 if hs[i] == min(hs) else 1 if hs[i] == max(hs) else 0,
                                 -1 if ws[i] == min(ws) else 1 if ws[i] == max(ws) else 0,
                                 -1 if xs[i] == min(xs) else 1 if xs[i] == max(xs) else 0,
                                 -1 if zs[i] == min(zs) else 1 if zs[i] == max(zs) else 0,
                                 0.]
                                )

        # add relations to all others
        for j in range(i+1, len(scene.objects)):
            relations[i, j, :] = array([compare_size(o, annot_objects[j]),
                                     compare_height(o, annot_objects[j]),
                                     compare_width(o, annot_objects[j]),
                                     compare_x(o, annot_objects[j]),
                                     compare_z(o, annot_objects[j], scene.depth),
                                     compare_distance(o, annot_objects[j], scene.depth)]
                                     )
            # all elements are anti-symmetric
            relations[j, i, :] = - relations[i, j, :]

        # distance is symmetric
        relations[:, :, -1] = abs(relations[:, :, -1])


    return SceneGraph(nodes=annot_objects, edges=relations)