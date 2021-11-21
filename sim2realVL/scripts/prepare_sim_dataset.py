from ..types import *
from ..data.sim_dataset import get_sim_rgbd_scenes_annotated
from ..utils.word_embedder import make_word_embedder
from ..models.visual_embedder import make_visual_embedder
from ..utils.image_proc import crop_box, crop_contour

import torch
from random import shuffle
from tqdm import tqdm



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
