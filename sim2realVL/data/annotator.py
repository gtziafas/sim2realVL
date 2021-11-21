from ..types import *
from .scene_parser import SceneParser
from ..utils.image_proc import *
from ..data.sim_dataset import SimScenesDataset

import pandas as pd 
from collections import Counter
from random import choice, random
from tqdm import tqdm


def annotate_dataset(dataset: SimScenesDataset, write_file: str):
    parser = SceneParser()

    all_queries, all_truths = [], []
    for scene in tqdm(dataset):
        graph = parser(scene)
        queries, truths = [], []
        categories = set([o.category for o in graph.nodes])
        category_count = Counter([o.category for o in graph.nodes])
        ambiguous = [i for i, o in enumerate(graph.nodes) if category_count[o.category] > 1]     
        
        for cat in categories:
            # multi-label annotations for ambigious queries
            idces = [i for i, o in enumerate(graph.nodes) if o.category == cat]
            queries.append(cat)
            truths.append(idces)
            if len(idces) == 1:
                continue

            # additional annotations for ambigious objects  
            for iteration, i in enumerate(idces):
                obj = graph.nodes[i]

                # add one color annot per ambiguous object
                queries.append(' '.join([obj.color, cat]))
                truths.append([i])

                # add one absolute spatial annot depending
                # on other ambiguous objects
                if not iteration:
                    most_left, most_right, closest, furthest = [i] * 4
                else:
                    most_left = i if obj.position_2d[0] < graph.nodes[most_left].position_2d[0] else most_left
                    most_right = i if obj.position_2d[0] > graph.nodes[most_right].position_2d[0] else most_right
                    closest = i if obj.position_2d[1] < graph.nodes[closest].position_2d[1] else closest
                    furthest = i if obj.position_2d[1] > graph.nodes[furthest].position_2d[1] else furthest

                # and up to three relative spatial annots 
                # behind/in-front, next to, left/right
                x_vector = graph.edges[i, :, 0]
                y_vector = graph.edges[i, :, 1]
                d_vector = graph.edges[i, :, 2]
                SOURCE = obj.category          

                proposed_queries = []
                # search for behind / in front of :
                if 1 in y_vector or -1 in y_vector:
                    target_obj_idx = choice(np.where(np.bitwise_or(y_vector==1, y_vector==-1))[0])
                    SPT = "behind" if y_vector[target_obj_idx] == -1 else "in front of"
                    target_obj = graph.nodes[target_obj_idx]
                    if target_obj_idx in ambiguous:
                        TARGET = ' '.join([target_obj.color, target_obj.category])
                    
                    else:
                        TARGET = target_obj.category

                    proposed_queries.append((' '.join([SOURCE, SPT, TARGET]), [i]))
                
                # search for next to (dist < threshold):
                if d_vector.min() <= 100:
                    target_obj_idx = np.argmin(d_vector)
                    SPT = "next to"
                    target_obj = graph.nodes[target_obj_idx]
                    if target_obj_idx in ambiguous:
                        TARGET = ' '.join([target_obj.color, target_obj.category])  
                    else:
                        TARGET = target_obj.category

                    proposed_queries.append((' '.join([SOURCE, SPT, TARGET]), [i]))


                # add a left / right
                # @todo: replace choice with the closest in x-direction
                target_obj_idx = choice(np.where(np.bitwise_or(x_vector==1, x_vector==-1))[0])
                SPT = "left from" if x_vector[target_obj_idx] == -1 else "right from"
                target_obj = graph.nodes[target_obj_idx]
                if target_obj_idx in ambiguous:
                    TARGET = ' '.join([target_obj.color, target_obj.category])
                
                else:
                    TARGET = target_obj.category

                # check if another ambiguous object of the same category qualifies for this query
                multi = [ii for ii in ambiguous if ii != target_obj_idx \
                                                 and graph.nodes[ii].category == cat \
                                                 and graph.edges[ii, target_obj_idx, 0] == x_vector[target_obj_idx]]

                proposed_queries.append((' '.join([SOURCE, SPT, TARGET]), multi))

                # check if query already given and append truths (2nd order ambiguity)
                for q, t in proposed_queries:
                    if q in queries:
                        query_idx = queries.index(q)
                        truths[query_idx] += t
                    else:
                        queries.append(q)
                        truths.append(t)

            # add absolute spatial per ambiguous category
            queries.append(' '.join(["left", cat]))
            queries.append(' '.join(["right", cat]))
            queries.append(' '.join(["closest", cat]))
            queries.append(' '.join(["furthest", cat]))
            truths.extend([[most_left], [most_right], [closest], [furthest]])

        all_truths.append([','.join(list(map(str, ts))) for ts in truths])
        all_queries.append(queries)

    # add annotations to existing table and save
    table = pd.DataFrame()
    table["image_id"] = [s.image_id for s in dataset]
    table["queries"] = all_queries
    table["truths"] = all_truths
    table.to_csv(write_file, sep = '\t')

