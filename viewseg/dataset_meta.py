# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


HYPERSIM_SEM_SEG_CATEGORIES = [
    "wall", "floor", "cabinet", "bed", "chair", 
    "sofa", "table", "door", "window", "bookshelf", 
    "picture", "counter", "blinds", "desk", "shelves", 
    "curtain", "dresser", "pillow", "mirror", "floor mat", 
    "clothes", "ceiling", "books", "refridgerator", "television", 
    "paper", "towel", "shower curtain", "box", "whiteboard", 
    "person", "night stand", "toilet", "sink", "lamp", 
    "bathtub", "bag", #"otherstructure", "otherfurniture", "otherprop",
]

NYU40_CATEGORIES = [
    {"color": [200, 236, 162], "isthing": 0, "id": 1, "name": "wall"},
    {"color": [154, 172, 224], "isthing": 0, "id": 2, "name": "floor"},
    {"color": [126, 103, 164], "isthing": 1, "id": 3, "name": "cabinet"},
    {"color": [241, 254, 112], "isthing": 1, "id": 4, "name": "bed"},
    {"color": [254, 192, 105], "isthing": 1, "id": 5, "name": "chair"},
    {"color": [241, 254, 112], "isthing": 1, "id": 6, "name": "sofa"},
    {"color": [255, 134, 117], "isthing": 1, "id": 7, "name": "table"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "door"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "window"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "bookshelf"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "picture"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "counter"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "blinds"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "desk"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "shelves"},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "curtain"},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "dresser"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "pillow"},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "mirror"},
    {"color": [120, 166, 157], "isthing": 1, "id": 20, "name": "floor mat"},
    {"color": [110, 76, 0], "isthing": 1, "id": 21, "name": "clothes"},
    {"color": [254, 236, 193], "isthing": 0, "id": 22, "name": "ceiling"},
    {"color": [199, 100, 0], "isthing": 1, "id": 23, "name": "books"},
    {"color": [72, 0, 118], "isthing": 1, "id": 24, "name": "refridgerator"},
    {"color": [255, 179, 240], "isthing": 1, "id": 25, "name": "television"},
    {"color": [0, 125, 92], "isthing": 1, "id": 26, "name": "paper"},
    {"color": [209, 0, 151], "isthing": 1, "id": 27, "name": "towel"},
    {"color": [188, 208, 182], "isthing": 1, "id": 28, "name": "shower curtain"},
    {"color": [0, 220, 176], "isthing": 1, "id": 29, "name": "box"},
    {"color": [255, 99, 164], "isthing": 1, "id": 30, "name": "whiteboard"},
    {"color": [92, 0, 73], "isthing": 1, "id": 31, "name": "person"},
    {"color": [133, 129, 255], "isthing": 1, "id": 32, "name": "night stand"},
    {"color": [78, 180, 255], "isthing": 1, "id": 33, "name": "toilet"},
    {"color": [0, 228, 0], "isthing": 1, "id": 34, "name": "sink"},
    {"color": [174, 255, 243], "isthing": 1, "id": 35, "name": "lamp"},
    {"color": [45, 89, 255], "isthing": 1, "id": 36, "name": "bathtub"},
    {"color": [134, 134, 103], "isthing": 1, "id": 37, "name": "bag"},
]


def register_all_hypersim(root):
    root = os.path.join(root, 'hypersim_sem_seg')
    for name, dirname in [("train", "train"), ("val", "val"), ("test", "test")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"hypersim_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        stuff_colors = [k["color"] for k in NYU40_CATEGORIES]
        MetadataCatalog.get(name).set(
            stuff_classes=HYPERSIM_SEM_SEG_CATEGORIES[:],
            stuff_colors=stuff_colors,
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


def register_all_replica(root):
    root = os.path.join(root, 'replica_sem_seg')
    for name, dirname in [("train", "train"), ("val", "val"), ("test", "test")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"replica_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        stuff_colors = [k["color"] for k in NYU40_CATEGORIES]
        MetadataCatalog.get(name).set(
            stuff_classes=HYPERSIM_SEM_SEG_CATEGORIES[:],
            stuff_colors=stuff_colors,
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

