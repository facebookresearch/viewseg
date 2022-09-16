# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import List, Optional, Tuple, Dict
import warnings
import h5py
import numpy as np
from PIL import Image
import torch
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset

from .dataset_meta import register_all_hypersim, register_all_replica


DEFAULT_DATA_ROOT = "/nfs/turbo/fouheyUnrep/syqian/viewseg_data/"
ALL_DATASETS = ("hypersim", "replica")


def collate_cameras(cameras_batch):
    """
    Create one batched cameras class based on the params from the batch 
    Currently this assumes the camera type is PerspectiveCameras.
    """
    Rs = torch.cat([c.R for c in cameras_batch], dim=0).view(-1, 3, 3)  # (N, 3, 3)
    Ts = torch.cat([c.T for c in cameras_batch], dim=0).view(-1, 3)  # (N, 3)
    focal_lengths = [c.focal_length for c in cameras_batch]
    focal_lengths = torch.cat(focal_lengths, dim=0)
    
    cameras = PerspectiveCameras(R=Rs, T=Ts, focal_length=focal_lengths)
    return cameras


def collate_fn(batch) -> Dict:
    """
    Create batched tensors/classes for all the elements in the batch
    """
    collated_dict = {}
    for k in batch[0].keys():
        if k in ["target_camera", "source_camera", "pair_id"]:
            continue
        collated_k = [d[k] for d in batch]

        if torch.is_tensor(collated_k[0]):
            collated_dict[k] = torch.stack(collated_k, dim=0)
        elif collated_k[0] is None:
            collated_dict[k] = None
        else:
            collated_dict[k] = torch.tensor(collated_k)
    
    # Collate params from all the camera classes
    collated_dict["source_camera"] = collate_cameras([b["source_camera"] for b in batch])
    collated_dict["target_camera"] = collate_cameras([b["target_camera"] for b in batch])
    collated_dict["pair_id"] = [b["pair_id"] for b in batch]
    return collated_dict


def prepare_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    rendering_name: str = None,
    autodownload: bool = False,
    num_views: int = 1,
    load_depth: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:

    dataset_dir = os.path.join(data_root, dataset_name)
    rendering_dir = os.path.join(data_root, rendering_name)
    dataset = {}

    for split in ['train', 'val', 'test']:
        cameras_path = os.path.join(dataset_dir, 'cameras_{}.pt'.format(split))
        train_data = torch.load(cameras_path)
        entries = train_data
        for entry in entries:
            for view_property in ['source', 'target']:
                for view_entry in entry[view_property]:
                    view_cam = PerspectiveCameras(
                        focal_length=[view_entry['focal_length']],
                        R=[view_entry['R']],
                        T=[view_entry['T']],
                    )
                    view_entry['camera'] = view_cam

        dataset[split] = ViewSegDataset(
            dataset_name,
            rendering_dir,
            entries, 
            image_size, 
            num_views=num_views, 
            load_depth=load_depth
        )


    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    return train_dataset, val_dataset, test_dataset



def get_viewseg_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = False,
    num_views: int = 1,
    load_depth: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get ViewSeg datasets.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.
        autodownload: Auto-download the dataset files in case they are missing.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    # metadata
    register_all_hypersim(DEFAULT_DATA_ROOT)
    register_all_replica(DEFAULT_DATA_ROOT)

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    if autodownload:
        raise NotImplementedError("autodownloading viewseg-style datasets is not supported for now.")

    rendering_name = None
    if dataset_name.startswith('hypersim'):
        rendering_name = 'hypersim_renderings'

    return prepare_datasets(dataset_name, image_size, data_root, rendering_name, autodownload, num_views, load_depth)


class ViewSegDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str,
        entries: List, 
        image_size: Tuple, 
        num_views: int = 1,
        load_depth: bool = False,
    ):
        """
        Args:
            entries: The list of dataset entries.
        """
        self._dataset_name = dataset_name
        self._dataset_dir = dataset_dir
        self._entries = entries
        self._image_size = image_size
        self._num_views = num_views
        self._load_depth = load_depth

    def __len__(self):
        return len(self._entries)

    def load_entry(self, entry):
        # read img and semantic annotation
        image_size = self._image_size
        img_path = os.path.join(self._dataset_dir, entry['img_path'])
        sem_path = os.path.join(self._dataset_dir, entry['sem_path'])
        scene_id = entry['scene_id']
        if self._load_depth:
            meters_per_asset = entry['meters_per_asset']
            depth_path = os.path.join(self._dataset_dir, entry['depth_path'])

        # read image
        image = torch.FloatTensor(np.array(Image.open(img_path))) / 255.0

        # read semantic segmentation
        if sem_path is None:
            warnings.warn("semantics not available. prepare pseudo semantics.")
            sem_label = np.zeros_like(image).astype(int)[:, :, 0]
        elif sem_path.endswith('npy'):  # replica
            sem_label = np.load(sem_path)
        elif sem_path.endswith('hdf5'):  # hypersim
            sem_label = h5py.File(sem_path, "r").get("dataset")
            sem_label = np.array(sem_label, dtype=int)
        else:
            raise NotImplementedError("cannot recognize {}".format(sem_path))
        sem_label = torch.LongTensor(sem_label)

        # ignore otherstructure
        # we map Replica classes to hypersim manually, so only hypersim has
        # classes such as otherstructure.
        if self._dataset_name.startswith('hypersim'):
            sem_label -= 1
            sem_label[sem_label >= 37] = -1 # 37, 38, 39 
            sem_label[sem_label < 0] = -1

        # read depth
        if self._load_depth:
            try:
                if depth_path.endswith('hdf5'):
                    depth = h5py.File(depth_path, "r").get("dataset")
                    depth = np.array(depth, dtype=np.float32)
                    depth = hypersim_distance_to_depth(depth)
                    depth = torch.FloatTensor(depth)
                elif depth_path.endswith('npy'):
                    depth = np.load(depth_path)
                    depth = torch.FloatTensor(depth)
            except:
                warnings.warn("depth not available. prepare pseudo depth")
                depth = np.ones_like(sem_label).astype(np.float32)
                depth = torch.FloatTensor(depth)
        else:
            depth = None

        # resize images to image_size
        scale_factors = [s_new / s for s, s_new in zip(image.shape[0:2], image_size)]
        if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
            raise ValueError(
                "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
            )
        scale_factor = sum(scale_factors) * 0.5
        if scale_factor != 1.0:
            image = torch.nn.functional.interpolate(
                image.permute(2, 0, 1).unsqueeze(0),
                size=tuple(image_size),
                mode="bilinear",
            )[0].permute(1, 2, 0)

            # shape: (H, W) -> (H, W, 1)
            sem_label = sem_label.unsqueeze(2).float()
            sem_label = torch.nn.functional.interpolate(
                sem_label.permute(2, 0, 1).unsqueeze(0),
                size=tuple(image_size),
                mode="nearest",
            )[0].permute(1, 2, 0)
            sem_label = sem_label[:, :, 0].long()

            if self._load_depth:
                depth = depth.unsqueeze(2).float()
                depth = torch.nn.functional.interpolate(
                    depth.permute(2, 0, 1).unsqueeze(0),
                    size=tuple(image_size),
                    mode="nearest",
                )[0].permute(1, 2, 0)
                depth = depth[:, :, 0]
    
        ret_entry = {
            'image': image,
            'sem_label': sem_label,
            'depth': depth,
            'camera': entry['camera'],
            'camera_idx': entry['camera_idx'],
        }
        return ret_entry
    

    def __getitem__(self, index):
        """
        Sample a source camera and image and a target camera and image
        """
        # If there are multiple source views, concat them
        sources = [self.load_entry(source) for source in self._entries[index]['source']]
        if self._num_views > 1:
            # only load first self._num_views
            sources = sources[:self._num_views]

            # collate entries
            cameras = collate_cameras([s["camera"] for s in sources])
            sem_labels = torch.stack([s["sem_label"] for s in sources], dim=0)
            if self._load_depth:
                depths = torch.stack([s["depth"] for s in sources], dim=0)
            images = torch.cat([s["image"][None] for s in sources], dim=0)
            camera_idxs = torch.tensor([s["camera_idx"] for s in sources], dtype=torch.int64)

            source = {
                "image": images, 
                "sem_label": sem_labels, 
                "camera": cameras, 
                "camera_idx": camera_idxs
            }
            if self._load_depth:
                source['depth'] = depths
            else:
                source['depth'] = None
        else:
            source = sources[0]

        # there is only one target view
        target = self._entries[index]['target'][0]
        target = self.load_entry(target)

        pair_id = self._entries[index]['pair_id']

        source = {"source_%s" % k: v for k, v in source.items()}
        target = {"target_%s" % k: v for k, v in target.items()}
        
        target.update(source)
        target['pair_id'] = pair_id

        return target


def hypersim_distance_to_depth(distance):
    """
    hypersim depth is actually the distance from the camera.
    here we convert them to depth
    see https://github.com/apple/ml-hypersim/issues/9
    """
    intWidth, intHeight = 1024, 768
    fltFocal = 886.81
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    depth = distance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return depth
