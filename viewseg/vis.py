# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import List, Dict
import pdb
import numpy as np
import os
from PIL import Image
import cv2

import torch
from pytorch3d.renderer import ray_bundle_to_ray_points, RayBundle
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer


def get_first_camera(cameras):
    """
    Get the first camera in the batch
    """
    R = cameras.R[0, ...]
    T = cameras.T[0, ...]
    K = cameras.get_projection_transform().get_matrix()[0, ...]
    new_cam = cameras.__class__(R=R[None], T=T[None], K=K[None])
    return new_cam

def process_output_cache(output_cache: List[Dict]):
    """
    Return only the first elem of each batch for visualization
    """
    new_output_cache = []
    for o in output_cache:
        o["source_image"] = o["source_image"][0, ...]
        o["image"] = o["image"][0, ...]
        o["camera"] = get_first_camera(o["camera"])
        if "coarse_ray_bundle" in o:
            bundle = o["coarse_ray_bundle"]
            o["coarse_ray_bundle"] = RayBundle(
                origins=bundle.origins[0, ...],
                directions=bundle.directions[0, ...],
                lengths=bundle.lengths[0, ...],
                xys = bundle.xys[0, ...]
            )
        new_output_cache.append(o)
    return new_output_cache
    
def visualize_nerf_outputs(
    nerf_out: dict, output_cache: List, viz: Visdom, visdom_env: str, metadata,
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        output_cache: A list with outputs of several training render passes.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
    """
    print("Plotting to visdom env: ", visdom_env)

    # Show the training images.
    # If batched pick the first image in the batch
    src_ims = torch.stack([o["source_image"][0] for o in output_cache])   
    tgt_ims = torch.stack([o["image"][0] for o in output_cache])
    N, H, W, _ = tgt_ims.shape

    # Reshape if there are multiple source views
    if src_ims.ndim == 5:
        NS = src_ims.shape[1]
        tgt_ims = tgt_ims[:, None]
        im_join = torch.cat([src_ims, tgt_ims], dim=1)
        im_join = im_join.permute(1, 0, 2, 3, 4)
        ims = im_join.reshape(-1, H, W, 3)
    elif src_ims.ndim == 4:  # single source view
        src_ims = src_ims[:, None]
        tgt_ims = tgt_ims[:, None]
        im_join = torch.cat([src_ims, tgt_ims], dim=1)
        ims = im_join.reshape(-1, H, W, 3)
    else:
        raise ValueError("incorrect src_ims shape")

    viz.images(
        ims.permute(0, 3, 1, 2),  # (B, 3, H, W)
        nrow=len(tgt_ims), # (show each source example with target examples below)
        env=visdom_env,
        win="images",
        opts={"title": "train images: source (top) | target (bottom)"}
    )

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    if "source_rgb_coarse" in nerf_out:
        source_ims_full = torch.cat(
            [
                nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
                for imvar in ("source_rgb_coarse", "source_rgb_fine", "source_rgb_gt")
            ],
            dim=2,
        )
        ims_full = torch.cat([source_ims_full, ims_full], dim=2)

    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "source_coarse | source_fine | source_gt | target_coarse | target_fine | target_gt"},
    )

    # Show the coarse and fine semantics together with the ground truth images.
    semantics_full = []
    if "source_semantic_coarse" in nerf_out:
        for imvar in ("source_semantic_coarse", "source_semantic_fine", "source_semantic_gt"):
            if imvar != "source_semantic_gt": # [minibatch, 320, 320, 102]
                sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().argmax(dim=0)
            else: # [1, 320, 320, 1]
                sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()[0]

            # convert to detectron2 convention
            # viewseg: 0 ~ n_classes, -1 for invalid pixels
            # detectron2: 0 ~ n_classes, 255 for invalid pixels
            sem[sem < 0] = 255

            image = nerf_out["rgb_gt"][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            image = image.permute(1, 2, 0)
            image = (image * 255.0).numpy().astype(np.uint8)
            instance_mode = ColorMode.IMAGE
            visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
            vis_output = visualizer.draw_sem_seg(sem)
            semantic_img = vis_output.get_image() 

            semantic_img = torch.FloatTensor(semantic_img / 255.0)
            semantic_img = semantic_img.permute(2, 0, 1)
            semantics_full.append(semantic_img)

    for imvar in ("semantic_coarse", "semantic_fine", "semantic_gt"):
        if nerf_out[imvar] is None:
            sem = torch.zeros_like(nerf_out["semantic_coarse"][0].permute(2, 0, 1).detach().cpu()[0])
        if imvar != "semantic_gt": # [minibatch, 320, 320, 102]
            sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().argmax(dim=0)
        else: # [1, 320, 320, 1]
            sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()[0]

        # convert to detectron2 convention
        # viewseg: 0 ~ n_classes, -1 for invalid pixels
        # detectron2: 0 ~ n_classes, 255 for invalid pixels
        sem[sem < 0] = 255

        image = nerf_out["rgb_gt"][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
        image = image.permute(1, 2, 0)
        image = (image * 255.0).numpy().astype(np.uint8)
        instance_mode = ColorMode.IMAGE
        visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
        vis_output = visualizer.draw_sem_seg(sem)
        semantic_img = vis_output.get_image() 

        semantic_img = torch.FloatTensor(semantic_img / 255.0)
        semantic_img = semantic_img.permute(2, 0, 1)
        semantics_full.append(semantic_img)

    semantics_full = torch.cat(semantics_full, dim=2)

    viz.image(
        semantics_full,
        env=visdom_env,
        win="semantics_full",
        opts={"title": "source_coarse | source_fine | source_gt | target_coarse | target_fine | target_gt"},
    )

    # Show depth
    if nerf_out['depth_fine'] is not None:
        depth_full = torch.cat(
            [
                nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 10.0) / 10.0
                for imvar in ("depth_coarse", "depth_fine", "depth_gt")
            ],
            dim=2,
        )
        if "source_depth_coarse" in nerf_out:
            source_depth_full = torch.cat(
                [
                    nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 10.0) / 10.0
                    for imvar in ("source_depth_coarse", "source_depth_fine", "source_depth_gt")
                ],
                dim=2,
            )
            depth_full = torch.cat([source_depth_full, depth_full], dim=2)

        viz.image(
            depth_full,
            env=visdom_env,
            win="depth_full",
            opts={"title": "source_coarse | source_fine | source_gt | target_coarse | target_fine | target_gt"},
        )

    # Make a 3D plot of training cameras and their emitted rays.
    if "coarse_ray_bundle" in output_cache[0]:
        camera_trace = {
            f"camera_{ci:03d}": o["camera"].cpu() for ci, o in enumerate(output_cache)
        }
        ray_pts_trace = {
            f"ray_pts_{ci:03d}": Pointclouds(
                ray_bundle_to_ray_points(o["coarse_ray_bundle"])
                .detach()
                .cpu()
                .view(1, -1, 3)
            )
            for ci, o in enumerate(output_cache)
        }
        plotly_plot = plot_scene(
            {
                "training_scene": {
                    **camera_trace,
                    **ray_pts_trace,
                },
            },
            pointcloud_max_points=5000,
            pointcloud_marker_size=1,
            camera_scale=0.3,
        )
        viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")


def save_nerf_outputs(
    export_dir: str, 
    batch_idx: int, 
    nerf_out: dict, 
    source_nerf_outs: List = [],
    metadata = None,
):
    """
    Save the outputs of the `RadianceFieldRenderer` to disk.

    Args:
        export_dir: save directory
        nerf_out: An output of the test rendering pass.
        source_nerf_out: 
    """
    # does nerf_out has depth?
    use_depth = False
    if nerf_out['depth_fine'] is not None:
        use_depth = True
    
    # target view
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    ims_full = ims_full.permute(1, 2, 0)
    frame_path = os.path.join(export_dir, f"{batch_idx:05d}_target_rgb.png")
    Image.fromarray((ims_full.numpy() * 255.0).astype(np.uint8)).save(frame_path)

    semantics_full = []
    for imvar in ("semantic_coarse", "semantic_fine", "semantic_gt"):
        if imvar != "semantic_gt": # [minibatch, height, width, # of classes]
            sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().argmax(dim=0)
        else: # [minibatch, height, width, 1]
            sem = nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()[0]

        # convert to detectron2 convention
        # viewseg: 0 ~ n_classes, -1 for invalid pixels
        # detectron2: 0 ~ n_classes, 255 for invalid pixels
        sem[sem < 0] = 255

        image = nerf_out["rgb_gt"][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
        image = image.permute(1, 2, 0)
        image = (image * 255.0).numpy().astype(np.uint8)
        instance_mode = ColorMode.IMAGE
        visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
        vis_output = visualizer.draw_sem_seg(sem)
        semantic_img = vis_output.get_image() 

        semantic_img = torch.FloatTensor(semantic_img / 255.0)
        semantic_img = semantic_img.permute(2, 0, 1)
        semantics_full.append(semantic_img)

    semantics_full = torch.cat(semantics_full, dim=2)
    semantics_full = semantics_full.permute(1, 2, 0)
    semantic_path = os.path.join(export_dir, f"{batch_idx:05d}_target_semantic.png")
    Image.fromarray((semantics_full.cpu().numpy() * 255.0).astype(np.uint8)).save(semantic_path)

    if use_depth:
        depth_full = torch.cat(
            [
                nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()
                for imvar in ("depth_coarse", "depth_fine", "depth_gt")
            ],
            dim=2,
        )

        depth_full = depth_full.permute(1, 2, 0)

        # scale depth for the best visualization
        depth_max = min(depth_full.max(), 20)
        depth_full[depth_full > depth_max] = depth_max
        depth_full = depth_full / depth_max * 255

        depth_color = cv2.applyColorMap(depth_full.numpy().astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
        depth_color[np.isnan(depth_color)] = 0
        frame_path = os.path.join(export_dir, f"{batch_idx:05d}_target_depth.png")
        Image.fromarray(depth_color).save(frame_path)

    # source views
    for source_idx, source_nerf_out in enumerate(source_nerf_outs):
        # rgb
        ims_full = torch.cat(
            [
                source_nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
                for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
            ],
            dim=2,
        )
        ims_full = ims_full.permute(1, 2, 0)
        frame_path = os.path.join(export_dir, f"{batch_idx:05d}_sources{source_idx:02d}_rgb.png")
        Image.fromarray((ims_full.numpy() * 255.0).astype(np.uint8)).save(frame_path)

        # semantic
        semantics_full = []
        for imvar in ("semantic_coarse", "semantic_fine", "semantic_gt"):
            if imvar != "semantic_gt": # [minibatch, height, width, # of classes]
                sem = source_nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().argmax(dim=0)
            else: # [minibatch, height, width, 1]
                sem = source_nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()[0]

            # convert to detectron2 convention
            # viewseg: 0 ~ n_classes, -1 for invalid pixels
            # detectron2: 0 ~ n_classes, 255 for invalid pixels
            sem[sem < 0] = 255

            image = source_nerf_out["rgb_gt"][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            image = image.permute(1, 2, 0)
            image = (image * 255.0).numpy().astype(np.uint8)
            instance_mode = ColorMode.IMAGE
            visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
            vis_output = visualizer.draw_sem_seg(sem)
            semantic_img = vis_output.get_image()
            semantic_img = torch.FloatTensor(semantic_img / 255.0)
            semantic_img = semantic_img.permute(2, 0, 1)
            semantics_full.append(semantic_img)
        semantics_full = torch.cat(semantics_full, dim=2)
        semantics_full = semantics_full.permute(1, 2, 0)
        semantic_path = os.path.join(export_dir, f"{batch_idx:05d}_sources{source_idx:02d}_semantic.png")
        Image.fromarray((semantics_full.cpu().numpy() * 255.0).astype(np.uint8)).save(semantic_path)

        # depth
        if use_depth:
            depth_full = torch.cat(
                [
                    source_nerf_out[imvar][0].permute(2, 0, 1).detach().cpu()
                    for imvar in ("depth_coarse", "depth_fine", "depth_gt")
                ],
                dim=2,
            )
            depth_full = depth_full.permute(1, 2, 0)

            # scale depth for the best visualization
            depth_max = min(depth_full.max(), 20)
            depth_full[depth_full > depth_max] = depth_max
            depth_full = depth_full / depth_max * 255

            depth_color = cv2.applyColorMap(depth_full.numpy().astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            depth_color[np.isnan(depth_color)] = 0
            frame_path = os.path.join(export_dir, f"{batch_idx:05d}_sources{source_idx:02d}_depth.png")
            Image.fromarray(depth_color).save(frame_path)

