# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import warnings
from typing import List, Optional, Tuple
from PIL import Image
from pathlib import Path
import math
import pdb
import collections
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from iopath.common.file_io import PathManager
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.io.mtl_io import load_mtl, make_mesh_texture_atlas
from pytorch3d.io.utils import _check_faces_indices, _make_tensor, _open_file
from pytorch3d.renderer import TexturesAtlas, TexturesUV
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.transforms import Rotate, RotateAxisAngle, Translate, Transform3d, quaternion_to_matrix


def single_gpu_prepare(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    del state_dict
    return new_state_dict

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])
    
def habitat_to_pytorch3d(
    R: torch.Tensor,
    T: torch.Tensor
):
    """
    Transforms the transforms from habitat world to pytorch3d world.
    Habitat world: +X right, +Y up, +Z from screen to us.
    Pytorch3d world: +X left, +Y up, +Z from us to screen.
    Compose the rotation by adding 180 rotation about the Y axis.
    """
    rotation = Rotate(R=R.transpose(0, 1))
    conversion = RotateAxisAngle(axis="y", angle=180)
    composed_transform = rotation.compose(conversion).get_matrix()
    composed_R = composed_transform[0, 0:3, 0:3]

    translation = Translate(x=T[None, ...])
    t_matrix = translation.compose(conversion).get_matrix()
    flipped_T = t_matrix[0, 3, :3]
    return composed_R, flipped_T


def generate_eval_video_cameras(
    train_dataset_entry,
    n_eval_cams: int = 100,
    trajectory_type: str = "figure_eight",
    trajectory_scale: float = 0.2,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Dataset[torch.Tensor]:
    """
    Generate a camera trajectory for visualizing a NeRF model.
    Args:
        train_dataset: The training dataset object.
        n_eval_cams: Number of cameras in the trajectory.
        trajectory_type: The type of the camera trajectory. Can be one of:
            circular: Rotating around the center of the scene at a fixed radius.
            figure_eight: Figure-of-8 trajectory around the center of the
                central camera of the training dataset.
            trefoil_knot: Same as 'figure_eight', but the trajectory has a shape
                of a trefoil knot (https://en.wikipedia.org/wiki/Trefoil_knot).
            figure_eight_knot: Same as 'figure_eight', but the trajectory has a shape
                of a figure-eight knot
                (https://en.wikipedia.org/wiki/Figure-eight_knot_(mathematics)).
        trajectory_scale: The extent of the trajectory.
        up: The "up" vector of the scene (=the normal of the scene floor).
            Active for the `trajectory_type="circular"`.
        scene_center: The center of the scene in world coordinates which all
            the cameras from the generated trajectory look at.
    Returns:
        Dictionary of camera instances which can be used as the test dataset
    """
    if trajectory_type in ("figure_eight", "trefoil_knot", "figure_eight_knot"):
        # cam_centers = torch.cat(
        #     [e["target_camera"].get_camera_center() for e in train_dataset]
        # )
        cam_centers = train_dataset_entry['target_camera'].get_camera_center()

        # get the nearest camera center to the mean of centers
        mean_camera_idx = (
            ((cam_centers - cam_centers.mean(dim=0)[None]) ** 2)
            .sum(dim=1)
            .min(dim=0)
            .indices
        )
        # generate the knot trajectory in canonical coords
        time = torch.linspace(0, 2 * math.pi, n_eval_cams + 1)[:n_eval_cams]
        if trajectory_type == "trefoil_knot":
            traj = _trefoil_knot(time)
        elif trajectory_type == "figure_eight_knot":
            traj = _figure_eight_knot(time)
        elif trajectory_type == "figure_eight":
            traj = _figure_eight(time)
        traj[:, 2] -= traj[:, 2].max()

        # transform the canonical knot to the coord frame of the mean camera
        traj_trans = (
            train_dataset_entry["target_camera"]
            .get_world_to_view_transform()
            .inverse()
        )
        #traj_trans = traj_trans.scale(cam_centers.std(dim=0).mean() * trajectory_scale)
        traj = traj_trans.transform_points(traj)

        scene_center = cam_centers.mean(dim=0).cpu().numpy().tolist()
        #scene_center = cam_centers[mean_camera_idx]

        # point all cameras towards the center of the scene
        R, T = look_at_view_transform(
            eye=traj,
            at=(scene_center,),  # (1, 3)
            up=(up,),  # (1, 3)
            device=traj.device,
        )
        R = R.cpu()
        T = T.cpu()

    elif trajectory_type == "circular":
        #cam_centers = torch.cat(
        #    [e["target_camera"].get_camera_center() for e in train_dataset]
        #)
        cam_centers = train_dataset_entry['source_camera'].get_camera_center()
        cam_centers = cam_centers.cuda()
        device = cam_centers.device

        # fit plane to the camera centers
        plane_mean = cam_centers.mean(dim=0)
        cam_centers_c = cam_centers - plane_mean[None]

        if up is not None:
            # us the up vector instead of the plane through the camera centers
            plane_normal = torch.FloatTensor(up)
            plane_normal = plane_normal.to(device)
        else:
            cov = (cam_centers_c.t() @ cam_centers_c) / cam_centers_c.shape[0]
            _, e_vec = torch.symeig(cov, eigenvectors=True)
            plane_normal = e_vec[:, 0]

        plane_dist = (plane_normal[None] * cam_centers_c).sum(dim=-1)
        cam_centers_on_plane = cam_centers_c - plane_dist[:, None] * plane_normal[None]

        cov = (
            cam_centers_on_plane.t() @ cam_centers_on_plane
        ) / cam_centers_on_plane.shape[0]
        _, e_vec = torch.symeig(cov, eigenvectors=True)
        traj_radius = (cam_centers_on_plane ** 2).sum(dim=1).sqrt().mean()
        angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams).to(device)
        traj = traj_radius * torch.stack(
            (torch.zeros_like(angle).to(device), angle.cos(), angle.sin()), dim=-1
        )
        traj = traj @ e_vec.t() + plane_mean[None]

        scene_center = cam_centers.mean(dim=0).cpu().numpy().tolist()
        scene_center[1] -= 1.0
        #scene_center = cam_centers[mean_camera_idx]

        pdb.set_trace()

        # point all cameras towards the center of the scene
        R, T = look_at_view_transform(
            eye=traj,
            at=(scene_center,),  # (1, 3)
            up=(up,),  # (1, 3)
            device=traj.device,
        )
        #pdb.set_trace()
        R = R.cpu()
        T = T.cpu()

    elif trajectory_type == "birdview":
        cam = train_dataset_entry['target_camera']
        dist = 5.0
        elev = torch.linspace(0, np.pi, n_eval_cams)
        scene_center = cam.unproject_points(torch.FloatTensor([[0, 0, dist]]))[0]

        plane_mean = scene_center# cam_centers.mean(dim=0)
        cam_centers_c = cam.get_camera_center().mean(dim=0) - plane_mean[None]

        if up is not None:
            # us the up vector instead of the plane through the camera centers
            plane_normal = torch.FloatTensor(up)
        else:
            cov = (cam_centers_c.t() @ cam_centers_c) / cam_centers_c.shape[0]
            _, e_vec = torch.symeig(cov, eigenvectors=True)
            plane_normal = e_vec[:, 0]

        plane_dist = (plane_normal[None] * cam_centers_c).sum(dim=-1)
        cam_centers_on_plane = cam_centers_c - plane_dist[:, None] * plane_normal[None]

        cov = (
            cam_centers_on_plane.t() @ cam_centers_on_plane
        ) / cam_centers_on_plane.shape[0]
        _, e_vec = torch.symeig(cov, eigenvectors=True)
        e_vec = - e_vec
        traj_radius = (cam_centers_on_plane ** 2).sum(dim=1).sqrt().mean()
        #angle = torch.linspace(0.5 * np.pi, 1.0 * np.pi, n_eval_cams)
        angle = torch.linspace(0.0 * np.pi, 2.0 * np.pi, n_eval_cams)
        traj = traj_radius * torch.stack(
            (torch.zeros_like(angle), angle.cos(), angle.sin()), dim=-1
        )
        traj = traj @ e_vec.t() + plane_mean[None]

        R, T = look_at_view_transform(
            eye=traj,
            at=(scene_center.numpy().tolist(),),  # (1, 3)
            up=(up,),  # (1, 3)
        )

        R = R[1:-1]
        T = T[1:-1]

        R = R.cpu()
        T = T.cpu()

    elif trajectory_type == "nerf_circle":
        target_camera = train_dataset_entry['target_camera']
        
        R = target_camera.R
        T = target_camera.T
        input_RT = torch.zeros([1, 4, 4])
        input_RT[:, :3, :3] = R
        input_RT[:, :3, 3] = T
        input_RT[:, 3, 3]= 1

        trans_factors = [0.75, 1.5]

        # input and output are b,4,4 rotation matrices
        output_RT = torch.zeros([n_eval_cams * len(trans_factors), 4, 4])

        for trans_id, trans_factor in enumerate(trans_factors):
            for i in range(n_eval_cams):
                # set output to same as input except 4th column which is 0,0,0,1 for now
                # remember column 1-3 is rotation, 4th column is position
                output_RT[:,:,:3] = input_RT[:,:,:3]
                output_RT[:,3,3] = 1
                
                # now, we set the position to the original position plus some scale times cyclical functions for x (horizontal), y (vertical), z (depth) 
                # scale down z for this case since model is better at handling horizontal or vertical movement
                num = i
                denom = n_eval_cams
                output_RT[trans_id * n_eval_cams + i, :3, 3] = input_RT[0, :3, 3] + trans_factor * torch.tensor([
                    np.sin(2 * np.pi * num / denom),
                    np.cos(2 * np.pi * num / denom),
                    0.4 * np.sin(2 * np.pi * (.25 + num / denom))
                ])
            
        R = output_RT[:, :3, :3]
        T = output_RT[:, :3, 3]
        R = R.cpu()
        T = T.cpu()

    else:
        raise ValueError(f"Unknown trajectory_type {trajectory_type}.")

    # get the average focal length and principal point
    #pdb.set_trace()
    """
    focal = torch.cat([e["target_camera"].focal_length for e in train_dataset]).mean(dim=0)
    p0 = torch.cat([e["target_camera"].principal_point for e in train_dataset]).mean(dim=0)

    source_images = [e["source_image"] for e in train_dataset]
    source_sem_labels = [e["source_sem_label"] for e in train_dataset]
    source_cameras = [e["source_camera"] for e in train_dataset]
    source_camera_ids = [e["source_camera_idx"] for e in train_dataset]
    source_depths = [e["source_depth"] for e in train_dataset]
    """
    focal = train_dataset_entry["target_camera"].focal_length
    p0 = train_dataset_entry["target_camera"].principal_point
    source_image = train_dataset_entry["source_image"]
    source_sem_label = train_dataset_entry["source_sem_label"]
    source_camera = train_dataset_entry["source_camera"]
    source_camera_idx = train_dataset_entry["source_camera_idx"]
    source_depth = train_dataset_entry["source_depth"]

    # assemble the dataset
    test_dataset = [
        {
            "target_image": None,
            "target_camera": PerspectiveCameras(
                focal_length=focal,
                principal_point=p0,
                R=R_[None],
                T=T_[None],
            ),
            "target_camera_idx": i,
            "target_sem_label": None,
            "target_depth": None,
            "source_image": source_image,
            "source_sem_label": source_sem_label,
            "source_camera": source_camera,
            "source_camera_idx": source_camera_idx,
            "source_depth": source_depth,
        }
        for i, (R_, T_) in enumerate(zip(R, T))
    ]

    return test_dataset


def _figure_eight_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = (2 + (2 * t).cos()) * (3 * t).cos()
    y = (2 + (2 * t).cos()) * (3 * t).sin()
    z = (4 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _trefoil_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = t.sin() + 2 * (2 * t).sin()
    y = t.cos() - 2 * (2 * t).cos()
    z = -(3 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _figure_eight(t: torch.Tensor, z_scale: float = 0.5):
    x = t.cos()
    y = (2 * t).sin() / 2
    z = t.sin() * z_scale
    return torch.stack((x, y, z), dim=-1)

def triangulate_pcd(sem_mask, semantic_pred, render_size, border_size):
    #verts = torch.ones(render_size).nonzero().numpy()
    verts = sem_mask.nonzero().numpy()
    vert_id_map = defaultdict(dict)
    for idx, vert in enumerate(verts):
        vert_id_map[vert[0]][vert[1]] = idx# + len(verts)

    height = render_size[0] - border_size * 2
    width = render_size[1] - border_size * 2

    semantic_pred = semantic_pred.numpy()

    triangles = []
    for vert in verts:
        # upper right triangle
        if (
            vert[0] < height - 1
            and vert[1] < width - 1
            and sem_mask[vert[0] + 1][vert[1] + 1]
            and sem_mask[vert[0]][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0]][vert[1] + 1]
        ):
            triangles.append(
                [
                    vert_id_map[vert[0]][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1] + 1],
                    vert_id_map[vert[0]][vert[1] + 1],
                ]
            )
        # bottom left triangle
        if (
            vert[0] < height - 1
            and vert[1] < width - 1
            and sem_mask[vert[0] + 1][vert[1] + 1]
            and sem_mask[vert[0]][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1]]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1] + 1]
        ):
            triangles.append(
                [
                    vert_id_map[vert[0]][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1] + 1],
                ]
            )
    triangles = np.array(triangles)
    triangles = torch.LongTensor(triangles)

    return triangles
