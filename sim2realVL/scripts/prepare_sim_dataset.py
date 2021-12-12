from ..types import *
from ..data.sim_dataset import *
from ..utils.word_embedder import make_word_embedder
from ..models.visual_embedder import make_visual_embedder
from ..utils.image_proc import crop_box, crop_contour

import torch
from random import shuffle, sample
from collections import Counter
from tqdm import tqdm



def count_annots_per_query(ds: List[AnnotatedScene], num_truths: int) -> List[int]:
    if num_truths == 1:
        q_ids, qs, ls = zip(*[(i, s.query, s.labels[s.truth]) for i, s in enumerate(ds) if isinstance(s.truth, int)])
    else:
        q_ids, qs, ls = zip(*[(i, s.query,  array(s.labels)[s.truth].tolist()) for i, s in enumerate(ds) if isinstance(s.truth, int)])

    _CQ, _CL = Counter(qs), Counter(ls)
    print(_CQ)
    print()
    print(_CL)
    return q_ids


def filter_samples_per_label(ds: List[AnnotatedScene],
    num_truths: int,
    size_per_label: int) -> Dict[str, List[int]]:
    if num_truths == 1:
        q_ids, qs, ls = zip(*[(i, s.query, [s.labels[s.truth]]) for i, s in enumerate(ds) if isinstance(s.truth, int)])
    else:
        q_ids, qs, ls = zip(*[(i, s.query, array(s.labels)[s.truth].tolist()) for i, s in enumerate(ds) if isinstance(s.truth, list) and len(s.truth) == num_truths])

    res = {}
    #print(set(sum(ls, [])))
    for label in set(sum(ls, [])):
        ids = [i for i, l in zip(q_ids, ls) if label in l]
        _size = min(size_per_label, len(ids))
        filtered_ids = sample(ids, _size)
        filtered_qs = [q for i, q in zip(q_ids, qs) if i in filtered_ids]
        filtered_ls = [l for i, l in zip(q_ids, ls) if i in filtered_ids]
        
        _CQ, _CL = Counter(filtered_qs), Counter(sum(filtered_ls, []))
        print(_CQ)
        print()
        print(_CL)
        print('==' * 48)

        res[label] = filtered_ids
    return res


def filter_samples_per_query(ds: List[AnnotatedScene],
    num_truths: int,
    size_per_query: int) -> Dict[str, List[int]]:
    if num_truths == 1:
        q_ids, qs, ls = zip(*[(i, s.query, [s.labels[s.truth]]) for i, s in enumerate(ds) if isinstance(s.truth, int)])
    else:
        q_ids, qs, ls = zip(*[(i, s.query,  array(s.labels)[s.truth].tolist()) for i, s in enumerate(ds) if isinstance(s.truth, list) and len(s.truth) == num_truths])

    res = {}
    #print(set(ls))
    for query in set(qs):
        ids = [i for i, q in zip(q_ids, qs) if q == query]
        _size = min(size_per_query, len(ids))
        filtered_ids = sample(ids, _size)
        filtered_qs = [q for i, q in zip(q_ids, qs) if i in filtered_ids]
        filtered_ls = [l for i, l in zip(q_ids, ls) if i in filtered_ids]
        
        _CQ, _CL = Counter(filtered_qs), Counter(sum(filtered_ls, []))
        print(_CQ)
        print()
        print(_CL)
        print('==' * 48)

        res[query] = filtered_ids
    return res


def prepare_dataset(ds: List[AnnotatedScene], 
                    image_loader: Callable[[int], array],
                    pretrained_features: bool, 
                    save: Maybe[str] = None,
                    order: str = "no"
                    ):
    H, W = 480, 640
    we = make_word_embedder()
    ve = make_visual_embedder()

    dataset = []
    covered_ids = {}
    for i, scene in enumerate(tqdm(ds)):
        # randomly permute objects if no order wanted
        truth = [scene.truth] if isinstance(scene.truth, int) else scene.truth
        truth = [1 if idx in truth else 0 for idx in range(len(scene.objects))]

        if order == "no":
            zipped = list(zip(scene.objects, truth))
            shuffle(zipped)
            objects, truth = zip(*zipped)
        elif order == "x":
            objects, truth = zip(*sorted(list(zip(scene.objects, truth)), key=lambda t: t[0].box.x))
        elif order == "y":
            objects, truth = zip(*sorted(list(zip(scene.objects, truth)), key=lambda t: t[0].box.y))
        # objects = scene.objects
            
        # dont do rendundant cropping
        if scene.image_id not in covered_ids:
            crops = [crop_contour(image_loader(scene.image_id), o.contour) for o in objects]
            feats = ve.features(crops) if pretrained_features else torch.stack(ve.tensorize(crops))
            covered_ids[scene.image_id] = feats
        else:
            feats = covered_ids[scene.image_id]
        
        position = torch.stack([torch.tensor([
            o.box.x, 
            o.box.y, 
            o.box.w, 
            o.box.h], dtype=longt) for o in objects])
        
        query = torch.tensor(we([scene.query])[0], dtype=floatt)

        truth = torch.tensor(truth, dtype=longt)
        # truth = torch.zeros(len(scene.objects), dtype=longt)
        # truth[scene.truth] = 1
        
        dataset.append((feats, query, truth, position))

    if save is not None:
        torch.save(dataset, save)

    return dataset


def get_tensorized_dataset(pretrained_features: bool, order: str, save_splits: bool):
    save = "checkpoints/SIM_dataset_{}_full.p".format("twostage" if pretrained_features else "onestage")
    ds = get_sim_rgbd_scenes_annotated()
    tensorized = prepare_dataset(
        [s for s in ds],
        ds.get_image_from_id,
        pretrained_features,
        save,
        order
        ) 

    if save_splits:
        print("Saving splits...")
        save = "checkpoints/SIM_dataset_{}_spatial_rel.p".format("twostage" if pretrained_features else "onestage")
        keywords = ['left', 'right', 'behind', 'front', 'next']
        idces = []
        for i, scene in enumerate(ds):
            tokens = scene.query.split()
            if len(tokens) > 3 or (len(tokens) == 3 and tokens[1] in keywords):
                idces.append(i)
        torch.save([s for i, s in enumerate(tensorized) if i in idces], save)

        save = "checkpoints/SIM_dataset_{}_spatial_abs.p".format("twostage" if pretrained_features else "onestage")
        keywords = ['left', 'right', 'behind', 'closest', 'furthest']
        idces = [i for i, s in enumerate(ds) if s.query.split()[0] in keywords]
        torch.save([s for i, s in enumerate(tensorized) if i in idces], save)

        save = "checkpoints/SIM_dataset_{}_categories.p".format("twostage" if pretrained_features else "onestage")
        cats = set(sum([s.categories for s in ds], []))
        idces = [i for i, s in enumerate(ds) if s.query in cats]
        torch.save([s for i, s in enumerate(tensorized) if i in idces], save)

        save = "checkpoints/SIM_dataset_{}_colors.p".format("twostage" if pretrained_features else "onestage")
        colors = ['blue', 'green', 'white', 'black', 'red', 'purple', 'brown', 'orange', 'yellow']
        idces = [i for i, s in enumerate(ds) if s.query.split()[0] in colors]
        torch.save([s for i, s in enumerate(tensorized) if i in idces], save)

    return tensorized


def annotate_word_tags(ds: List[AnnotatedScene]):
    colors = ['blue', 'green', 'white', 'black', 'red', 'purple', 'brown', 'orange', 'yellow']
    all_subj_words = set(sum([s.categories for s in ds], [])) 
    all_subj_words = sum([c.split() for c in all_subj_words], []) + colors

    def transform(split: str) -> Map[str, List[str]]:
        if split == "category" or split == "color":
            return lambda q: ['<subj>'] * len(q.split()) 

        elif split == "spatial_abs":
            return lambda q: ['<loc>'] + ['<subj>'] * (len(q.split()) - 1)

        elif split == "spatial_rel":
            def _transform_spt_rel(q: str) -> List[str]:
                tokens = q.split()
                is_rel = [0 if t in all_subj_words else 1 for t in tokens]
                rel_starts, rel_ends = is_rel.index(1), (len(is_rel) - is_rel[::-1].index(1))
                return ['<subj>'] * rel_starts + ['<rel>'] * (rel_ends - rel_starts) + ['<obj>'] * (len(is_rel) - rel_ends) 
            return _transform_spt_rel

        else:
            raise ValueError(f"unknown split {split}")
            
    queries = ds.queries
    annots = [None] * len(queries)
    for split in ["category", "color", "spatial_abs", "spatial_rel"]:
        tmp_indices = get_split_indices(ds, split)
        for index in tmp_indices:
            annots[index] = split
    assert None not in annots

    return [transform(annot)(query) for query, annot in zip(queries, annots)]


def get_dataset_word_tags():
    ds = get_sim_rgbd_scenes_annotated()
    return annotate_word_tags(ds)