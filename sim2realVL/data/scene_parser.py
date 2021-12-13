
from ..types import *
from ..utils.viz import render_graph

import numpy as np 


SIM_COLOR_MAP = {
    'mug_yellow'    : 'yellow',
    'cereal_box_1'  : 'purple',
    'cereal_box_2'  : 'yellow',
    'cereal_box_3'  : 'brown',
    'can_fanta'     : 'orange',
    'can_sprite'    : 'green',
    'can_coke'      : 'red',
    'can_pepsi'     : 'blue',
    'mug_green'     : 'green',
    'cap_red'       : 'red',
    'cap_white'     : 'white',
    'cap_black'     : 'black',
    'mug_red'       : 'red',
    'bowl_2'        : 'white',
    'bowl_1'        : 'red',
    'flashlight_yellow' : 'yellow',
    'flashlight_red'    : 'red',
    'flashlight_blue'   : 'blue',
    'flashlight_black'   : 'black'
}

SIM_SPECIAL_MAP = {
    'mug_yellow'    : 'cup,yellow cup',
    'cereal_box_1'  : 'Raisin Bran,Crunch',
    'cereal_box_2'  : 'Bran Flakes',
    'cereal_box_3'  : 'Chex,chocolate cereal',
    'can_fanta'     : 'fanta',
    'can_sprite'    : 'sprite',
    'can_coke'      : 'coke,cola,coca cola',
    'can_pepsi'     : 'pepsi',
    'mug_green'     : 'cup,green cup',
    'cap_red'       : 'hat,red hat',
    'cap_white'     : 'hat,white hat',
    'cap_black'     : 'hat,black hat',
    'mug_red'       : 'cup,red cup',
    'bowl_2'        : 'wide bowl,big bowl',
    'bowl_1'        : 'tall bowl',
    'flashlight_yellow' : 'torch,yellow torch',
    'flashlight_red'    : 'torch,red torch',
    'flashlight_blue'   : 'torch,blue torch',
    'flashlight_black'  : 'torch,black torch'
}
SIM_SPECIAL_MAP = {k: [s for s in v.split(',')] for k, v in SIM_SPECIAL_MAP.items()}


class SceneGraph(SceneGraph):

    def print(self):
        maxchars = 18
        header = ''.join([' '*maxchars] + [o.label + ' ' * (maxchars - len(o.label)) for o in self.nodes]) 
        rows = [['{' + str(int(row[0])) + ', ' +  str(int(row[1])) + ', ' +  str(round(row[2],3)) + '}' for row in self.edges[i,:,:].tolist()] for i in range(len(self.nodes))] 
        rows = [''.join([self.nodes[i].label + ' ' * (maxchars - len(self.nodes[i].label))] + [x + ' ' * (maxchars - len(x)) for x in rows[i]]) for i in range(len(rows))]
        print('\n'.join([header, *rows]))

    def render(self, path: str):
        with open('input.dot', 'w+') as f:
            items = ['item{} [ label="{}"];'.format(i+1, self.nodes[i].label) for i in range(len(self.nodes))]
            self_loops = ['item{} -> item{} [ label="{}"];'.format(i+1, i+1, self.nodes[i].color) for i in range(len(self.nodes))]
            edges = [['item{} -> item{} [ label={}];'.format(i+1, 1 + j, str(round(self.edges[i,j,-1],4))) for j in range(i+1,len(self.nodes))] for i in range(0,len(self.nodes))]
            
            f.write("digraph G {")
            f.write('\n')
            f.write('  ')
            f.write('\n  '.join([*items, *self_loops, *sum(edges, [])]))
            f.write('\n')
            f.write('}')
        render_graph('input.dot', path)

    def get_mask(self, query: str):
        if query == "left":
            x_mask = self.edges[:,:,0]
            return x_mask == -1

        elif query == "right":
            x_mask = self.edges[:,:,0]
            return x_mask == 1

        elif query == "behind":
            y_mask = self.edges[:,:,1]
            return y_mask == -1

        elif query == "front":
            y_mask = self.edges[:,:,1]
            return y_mask == 1 

        elif query == "closest":
            ys = np.array([o.position_2d[1] for o in self.nodes])
            ys = np.tile(ys, [ys.shape[0], 1])
            ys_pair = ys - ys.T
            return ys_pair > 0 

        elif query == "furthest":
            ys = np.array([o.position_2d[1] for o in self.nodes])
            ys = np.tile(ys, [ys.shape[0], 1])
            ys_pair = ys - ys.T
            return ys_pair < 0

        elif query == "next":
            d_mask = self.edges[:,:,2]
            return d_mask <= 100


class SceneParser(object):
    # parses an input scene into a graph containing semantic and spatial information about the objects
    # and their relations
    def __init__(self, resolution: Tuple[int, int] = (480, 640), use_2D_position: bool = False):
        self.img_height, self.img_width = resolution
        self.Y_threshold = self.img_height / 6
        self.use_2D_position = use_2D_position

    def __call__(self, scene: Scene) -> SceneGraph:
        colors, tags = zip(*[(SIM_COLOR_MAP[o.label], SIM_SPECIAL_MAP[o.label]) for o in scene.objects])
        # get all information for annotated object in graph's nodes
        nodes = [AnnotatedObjectSim(o.label, o.category, o.contour, o.box, o.rectangle, o.center_of_mass,
                o.position_2d, color=colors[i], special=tags[i]) for i, o in enumerate(scene.objects)]
        
        # get spatial relations between objects in graph's edges
        edges = np.empty((len(scene.objects), len(scene.objects), 3), dtype=float)

        lefts, rights, tops, bottoms = zip(*[(self.left(o.rectangle),
                                              self.right(o.rectangle),
                                              self.top(o.rectangle),
                                              self.bottom(o.rectangle)) for o in scene.objects])
        for i, o in enumerate(nodes):
            # add main diagonal relation element
            # edges[i, i] = array([-1 if self.left(o.rectangle) == min(lefts) else 1 if self.right(o.rectangle) == max(rights) else 0,
            #                      -1 if self.top(o.rectangle) == min(tops) else 1 if self.bottom(o.rectangle) == max(bottoms) else 0,
            #                      0.])
            edges[i, i] = array([0., 0., 1e5])

            # add relations to all others
            for j in range(i+1, len(scene.objects)):
                edges[i, j, :] = array([self.compare_x(o, nodes[j]), self.compare_y(o, nodes[j]), self.distance_2d(o, nodes[j])])
                
                # all elements are anti-symmetric
                edges[j, i, :] = - edges[i, j, :]

            # distance is symmetric
            edges[:, :, -1] = abs(edges[:, :, -1]) 

        return SceneGraph(nodes, edges)


    def compare_x(self, obj1: ObjectSim, obj2: ObjectSim) -> int:
        # -1 -> left, 0 -> nothing, 1-> right
        (x1, y1), (x2, y2) = obj1.center_of_mass, obj2.center_of_mass
        if x1 < x2:
            return 0 if self.right(obj1.rectangle) >= self.left(obj2.rectangle) else -1 # overlapping
        else:
            return 0 if self.right(obj2.rectangle) >= self.left(obj1.rectangle) else 1

    def compare_y(self, obj1: ObjectSim, obj2: ObjectSim) -> int:
        (x1, y1), (x2, y2) = obj1.center_of_mass, obj2.center_of_mass
        check_overlap = self.right(obj1.rectangle) >= self.left(obj2.rectangle) if x1 <= x2 else self.right(obj2.rectangle) >= self.left(obj1.rectangle)
        # if not overlapping in x, no y annotation (0)
        if not check_overlap:
            return 0
        # -1 -> behind, 0 -> nothing, 1 -> in front
        else:
            if y1 < y2:
                return -1 if self.top(obj2.rectangle) - self.bottom(obj1.rectangle) <= self.Y_threshold else 0
            else:
                return 1 if self.top(obj1.rectangle) - self.bottom(obj2.rectangle) <= self.Y_threshold else 0

    def left(self, rect: Rectangle) -> int:
        return min(rect.x1, rect.x2, rect.x3, rect.x4)

    def right(self, rect: Rectangle) -> int:
        return max(rect.x1, rect.x2, rect.x3, rect.x4)

    def top(self, rect: Rectangle) -> int:
        return min(rect.y1, rect.y2, rect.y3, rect.y4)

    def bottom(self, rect: Rectangle) -> int:
        return max(rect.y1, rect.y2, rect.y3, rect.y4)

    def distance_2d(self, obj1: ObjectSim, obj2: ObjectSim) -> float:
        if not self.use_2D_position:
            (x1, y1), (x2, y2) = obj1.center_of_mass, obj2.center_of_mass
        else:
            (x1, y1), (x2, y2) = obj1.position_2d, obj2.position_2d    
        return np.linalg.norm([y2 - y1, x2 - x1])
