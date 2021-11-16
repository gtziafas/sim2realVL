from .scene_parser import SceneParser
from ..data.sim_dataset import SimScenesDataset

import pandas as pd 


def annotate_dataset(dataset: SimScenesDataset, write_file: str):
    parser = SceneParser()
    all_queries, all_truths = [], []
    for scene in dataset:
        graph = parser(scene)
        queries, truths = [], []
        categories = set([o.category for o in graph.nodes])
        for cat in categories:
            # multi-label annotations for ambigious queries
            idces = [i for i, o in enumerate(graph.nodes) if o.category == cat]
            queries.append(cat)
            truths.append(idces)
            if len(idces) == 1:
                continue

            # additional annotations for ambigious objects
            for idx in idces:
                obj = graph.nodes[idx]
                queries.append(' '.join([obj.color, cat]))
                truths.append([idx])
        all_truths.append([','.join(list(map(str,ts))) for ts in truths])
        all_queries.append(queries)

    # add annotations to existing table and save
    table = pd.DataFrame()
    table["image_id"] = [s.image_id for s in dataset]
    table["queries"] = all_queries
    table["truths"] = all_truths
    table.to_csv(write_file, sep = '\t')

