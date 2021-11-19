from .scene_parser import SceneParser
from ..utils.image_proc import *
from ..data.sim_dataset import SimScenesDataset

import pandas as pd 
from collections import Counter
from random import choice
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
            for i in idces:
                obj = graph.nodes[i]

                # add one color annot per ambiguous object
                queries.append(' '.join([obj.color, cat]))
                truths.append([i])

                # and up to three spatial annots 
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

                    proposed_queries.append(' '.join([SOURCE, SPT, TARGET]))
                
                # search for next to (dist < threshold):
                if d_vector.min() <= 100:
                    target_obj_idx = np.argmin(d_vector)
                    SPT = "next to"
                    target_obj = graph.nodes[target_obj_idx]
                    if target_obj_idx in ambiguous:
                        TARGET = ' '.join([target_obj.color, target_obj.category])  
                    else:
                        TARGET = target_obj.category

                    proposed_queries.append(' '.join([SOURCE, SPT, TARGET]))


                # add a left / right
                target_obj_idx = choice(np.where(np.bitwise_or(x_vector==1, x_vector==-1))[0])
                SPT = "left from" if x_vector[target_obj_idx] == -1 else "right from"
                target_obj = graph.nodes[target_obj_idx]
                if target_obj_idx in ambiguous:
                    TARGET = ' '.join([target_obj.color, target_obj.category])
                
                else:
                    TARGET = target_obj.category

                proposed_queries.append(' '.join([SOURCE, SPT, TARGET]))

                # check if query already given and append truths (2nd order ambiguity)
                for q in proposed_queries:
                    if q in queries:
                        query_idx = queries.index(q)
                        truths[query_idx] += [i]
                    else:
                        queries.append(proposed_query)
                        truths.append([i])


        # spatial
        category_count = Counter([o.category for o in graph.nodes])
        ambiguous = [i for i, o in enumerate(graph.nodes) if category_count[o.category] > 1]     
        for i, obj in enumerate(graph.nodes):
            x_vector = graph.edges[i, :, 0]
            y_vector = graph.edges[i, :, 1]
            d_vector = graph.edges[i, :, 2]
            SOURCE = obj.category

            # search for behind / in front of :
            if 1 in y_vector or -1 in y_vector:
                target_obj_idx = choice(np.where(np.bitwise_or(y_vector==1, y_vector==-1))[0])
                SPT = "behind" if y_vector[target_obj_idx] == -1 else "in front of"
                
            # search for next to (dist < threshold):
            elif d_vector.min() <= 100:
                target_obj_idx = np.argmin(d_vector)
                SPT = "next to"

            # if dealing with ambiguous sample, resolve ambiguity with closer to / further from
            # elif i in ambiguous:
            #     current_ambiguous = [ii for ii, o in graph.nodes if o.category == SOURCE]
            #     closest = [np.argmin(graph.edges[ii, :, 2]) for ii in current_ambiguous]
            #     # closest to ambiguous objects dont match
            #     if len(closest) == len(set(closest)):
            #         target_obj_idx = np.argmin(d_vector)
            #         SPT = "closer to"

            # go to left/right
            else:
                target_obj_idx = choice(np.where(np.bitwise_or(x_vector==1, x_vector==-1))[0])
                SPT = "left from" if x_vector[target_obj_idx] == -1 else "right from"

            # resolve ambiguity if target object is ambigious
            target_obj = graph.nodes[target_obj_idx]
            if target_obj_idx in ambiguous:
                TARGET = ' '.join([target_obj.color, target_obj.category])
            
            else:
                TARGET = target_obj.category

            proposed_query = ' '.join([SOURCE, SPT, TARGET])
            
            # check if query already given and append truths (2nd order ambiguity)
            if proposed_query in queries:
                query_idx = queries.index(proposed_query)
                truths[query_idx] += [i]
                continue

            queries.append(proposed_query)
            truths.append([i])


        all_truths.append([','.join(list(map(str, ts))) for ts in truths])
        all_queries.append(queries)

    # add annotations to existing table and save
    table = pd.DataFrame()
    table["image_id"] = [s.image_id for s in dataset]
    table["queries"] = all_queries
    table["truths"] = all_truths
    table.to_csv(write_file, sep = '\t')

