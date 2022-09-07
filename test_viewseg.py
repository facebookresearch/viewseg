#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import warnings
import numpy as np
import random
from tqdm import tqdm
import collections
from omegaconf import DictConfig
from PIL import Image
import pdb
import imageio
import pickle
import cv2
import trimesh

import torch
import hydra
import pytorch3d
from pytorch3d.io import save_obj
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import RotateAxisAngle
from detectron2.data import MetadataCatalog

from viewseg.dataset import collate_fn, get_viewseg_datasets
from viewseg.renderer import SemanticRadianceFieldRenderer
from viewseg.vis import visualize_nerf_outputs, save_nerf_outputs
from viewseg.nerf.stats import Stats
from viewseg.encoder import build_spatial_encoder
from viewseg.utils import generate_eval_video_cameras, triangulate_pcd, single_gpu_prepare


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
# compatible with old training history 
sys.modules['panonerf'] = sys.modules['viewseg']


def export_mesh(cfg, test_dataloader, model, colormap, export_dir, device):
    """
    Construct 3D semantic mesh from predicted depth and 2D semantics. 
    """
    for batch_idx, test_batch in enumerate(tqdm(test_dataloader)):
        image = test_batch['target_image']
        sem_label = test_batch['target_sem_label']
        camera = test_batch['target_camera']
        depth = test_batch['target_depth']
        source_image = test_batch['source_image']
        source_camera = test_batch['source_camera']
        source_sem_label = test_batch['source_sem_label']
        source_depth = test_batch['source_depth']

        if image is not None:
            image = image.to(device)
        camera = camera.to(device)
        if depth is not None:
            depth = depth.to(device)
        if sem_label is not None:
            sem_label = sem_label.to(device)

        source_image = source_image.to(device)
        source_camera = source_camera.to(device)
        if source_depth is not None:
            source_depth = source_depth.to(device)
        source_sem_label = source_sem_label.to(device)

        # Activate eval mode of the model (lets us do a full rendering pass).
        model.eval()
        with torch.no_grad():
            test_nerf_out, _ = model(
                camera_hash=None,  # we do not use pre-cached cameras
                camera=camera,
                image=image,
                depth=depth,
                sem_label=sem_label,
                source_camera=source_camera,
                source_image=source_image,
            )

        export_path = os.path.join(export_dir, f"{batch_idx:04d}")
        os.makedirs(export_path, exist_ok=True)

        border_size = cfg.test.pcd_border_crop

        # rgb and semantic prediction
        rgb_pred = test_nerf_out['rgb_fine'].cpu()
        semantic_pred_raw = test_nerf_out['semantic_fine'][0].cpu()
        #rgb_pred = rgb_pred[0, border_size:-border_size, border_size:-border_size]
        semantic_pred_uncrop = semantic_pred_raw.argmax(dim=-1)
        semantic_pred_raw = semantic_pred_raw[border_size:-border_size, border_size:-border_size]

        semantic_pred_raw = torch.softmax(semantic_pred_raw, dim=2)
        semantic_pred_raw = torch.cat((semantic_pred_raw, 
            torch.ones((semantic_pred_raw.shape[0], semantic_pred_raw.shape[1], 1)) * cfg.test.pcd_confidence), dim=2
        )
        semantic_pred = semantic_pred_raw.argmax(dim=-1)
        sem_mask = torch.logical_not(semantic_pred == (semantic_pred_raw.shape[2] - 1))

        # pytorch3d NDC space
        # x: left
        # y: up
        # z: positive depth
        depth_pred = test_nerf_out['depth_fine'][0, :, :, 0]
        depth_pred[0:border_size, :] = 0.0
        depth_pred[-border_size:, :] = 0.0
        depth_pred[:, 0:border_size] = 0.0
        depth_pred[:, -border_size:] = 0.0

        # tqdm.write("gt_range: {} - {}".format(depth[valid_mask].min().cpu().item(), depth[valid_mask].max().cpu().item()))
        # tqdm.write("pred_range: {} - {}".format(depth_pred.min().cpu().item(), depth_pred.max().cpu().item()))

        render_size = (cfg.data.render_size[0], cfg.data.render_size[1])
        xys = depth_pred.nonzero().float()
        xys[:, 0] = - (xys[:, 0] / render_size[0] - 0.5) * 2
        xys[:, 1] = - (xys[:, 1] / render_size[1] - 0.5) * 2
        xys = xys.flip(1)
        depth_pred_flat = depth_pred[border_size:-border_size, border_size:-border_size].reshape(-1, 1)
        xy_depth = torch.cat([xys, depth_pred_flat], dim=1)
        #xy_depth = xy_depth[border_size:-border_size, border_size:-border_size]
        xy_depth = xy_depth[sem_mask.reshape(-1)]
        points = camera.unproject_points(xy_depth, world_coordinates=False)

        # convert points from PyTorch3D world to Habitat/Meshlab/Model-viewer world 
        # for the best visualization.
        # Habitat/model-viewer world: +X right, +Y up, +Z from screen to us.
        # Pytorch3d world: +X left, +Y up, +Z from us to screen.
        # Compose the rotation by adding 180 rotation about the Y axis.
        conversion = RotateAxisAngle(axis="y", angle=180).to(device)
        points = conversion.transform_points(points)
        points = points.cpu()

        # save results as obj, build faces
        verts = points
        faces = triangulate_pcd(sem_mask, semantic_pred, render_size, border_size)

        # obj format texture coordinate system
        verts_uvs = depth_pred.nonzero().float().cpu()
        verts_uvs[:, 0] = - verts_uvs[:, 0] / render_size[0]
        verts_uvs[:, 1] = - (1 - verts_uvs[:, 1] / render_size[1])
        verts_uvs = verts_uvs.flip(1)
        verts_uvs = verts_uvs[sem_mask.reshape(-1)]

        # PyTorch3D should support save_mesh with textures.
        # now let's use save_obj
        rgb_texture = rgb_pred[0]
        obj_path = os.path.join(export_path, 'pred.obj')

        save_obj(obj_path, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=rgb_texture)

        # save mesh with semantics as the texture
        sem_texture = colormap[semantic_pred_uncrop]
        sem_obj_path = os.path.join(export_path, 'pred_sem.obj')
        save_obj(sem_obj_path, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=sem_texture)

        # Convert obj to glb
        # pytorch3d does not support glb
        # model-viewer can only be run on glb
        glb_path = os.path.join(export_path, 'pred.glb')
        temp_mesh = trimesh.load(obj_path)
        temp_mesh.export(glb_path)

        sem_glb_path = os.path.join(export_path, 'pred_sem.glb')
        temp_mesh = trimesh.load(sem_obj_path)
        temp_mesh.export(sem_glb_path)

        # skip the loop if there are enough examples
        if batch_idx > cfg.test.num_samples and cfg.test.num_samples != -1:
            break


def export_video(cfg, test_dataloader, model, colormap, export_dir, device):
    for batch_idx, train_batch in enumerate(test_dataloader):
        test_dataset = generate_eval_video_cameras(
            train_batch,
            trajectory_type=cfg.test.trajectory_type,
            up=cfg.test.up,
            scene_center=cfg.test.scene_center,
            n_eval_cams=cfg.test.n_frames,
            trajectory_scale=cfg.test.trajectory_scale,
        )

        video_path = os.path.join(export_dir, f"scene_{batch_idx:05d}.mp4")
        writer = imageio.get_writer(video_path)
        

        for test_batch in tqdm(test_dataset):
            image = test_batch['target_image']
            sem_label = test_batch['target_sem_label']
            camera = test_batch['target_camera']
            depth = test_batch['target_depth']
            source_image = test_batch['source_image']
            source_camera = test_batch['source_camera']
            source_sem_label = test_batch['source_sem_label']
            source_depth = test_batch['source_depth']

            if image is not None:
                image = image.to(device)

            camera = camera.to(device)
            if depth is not None:
                depth = depth.to(device)
            if sem_label is not None:
                sem_label = sem_label.to(device)

            source_image = source_image.to(device)
            source_camera = source_camera.to(device)
            if source_depth is not None:
                source_depth = source_depth.to(device)
            source_sem_label = source_sem_label.to(device)


            # Activate eval mode of the model (lets us do a full rendering pass).
            model.eval()
            with torch.no_grad():
                test_nerf_out, test_metrics = model(
                    camera_hash=None,  # we do not use pre-cached cameras
                    camera=camera,
                    image=image,
                    depth=depth,
                    sem_label=sem_label,
                    source_camera=source_camera,
                    source_image=source_image,
                )


            # Store the video frame.
            frame = test_nerf_out["rgb_fine"][0].detach().cpu()
            frame = (frame.numpy() * 255.0).astype(np.uint8)
            
            sem = test_nerf_out["semantic_fine"][0].permute(2, 0, 1).detach().cpu().argmax(dim=0)
            semantic_img = (colormap[sem].numpy() * 255.0).astype(np.uint8)

            depth = test_nerf_out['depth_fine'][0].permute(2, 0, 1).detach().cpu()
            depth = depth.permute(1, 2, 0)
            depth[depth > 20.0] = 20.0
            depth = depth / 20.0 * 255
            depth = cv2.applyColorMap(depth.numpy().astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            depth[np.isnan(depth)] = 0

            frame = np.concatenate((frame, semantic_img, depth), axis=1)
            writer.append_data(frame)

        writer.close()
        
        # store target img
        source_image = source_image.cpu()[0]
        source_vis = torch.cat([img for img in source_image], dim=1)
        source_path = os.path.join(export_dir, f"scene_{batch_idx:05d}_source.png")
        Image.fromarray((source_vis * 255.0).numpy().astype(np.uint8)).save(source_path)

        # skip the loop if there are enough examples
        if batch_idx > cfg.test.num_samples and cfg.test.num_samples != -1:
            break

            
def export_imgs(cfg, test_dataloader, model, colormap, metadata, stats, export_dir, device):
    eval_stats = ["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"]
    stats = Stats(eval_stats)
    stats.new_epoch()

    # Run the main testing loop.
    for batch_idx, test_batch in enumerate(tqdm(test_dataloader)):
        image = test_batch['target_image']
        sem_label = test_batch['target_sem_label']
        camera = test_batch['target_camera']
        depth = test_batch['target_depth']
        source_image = test_batch['source_image']
        source_camera = test_batch['source_camera']
        source_sem_label = test_batch['source_sem_label']
        source_depth = test_batch['source_depth']

        if image is not None:
            image = image.to(device)
        camera = camera.to(device)
        if depth is not None:
            depth = depth.to(device)
        if sem_label is not None:
            sem_label = sem_label.to(device)

        source_image = source_image.to(device)
        source_camera = source_camera.to(device)
        if source_depth is not None:
            source_depth = source_depth.to(device)
        source_sem_label = source_sem_label.to(device)

        # Activate eval mode of the model (lets us do a full rendering pass).
        model.eval()
        with torch.no_grad():
            test_nerf_out, test_metrics = model(
                camera_hash=None,  # we do not use pre-cached cameras
                camera=camera,
                image=image,
                depth=depth,
                sem_label=sem_label,
                source_camera=source_camera,
                source_image=source_image,
                source_depth=source_depth,
            )

            source_nerf_outs = []

            if cfg.train.num_views > 1:
                for idx in range(cfg.train.num_views):
                    select_source_image = source_image[:, idx]
                    select_source_sem_label = source_sem_label[:, idx]
                    if source_depth is None:
                        select_source_depth = None
                    else:
                        select_source_depth = source_depth[:, idx]
                    select_R = source_camera.R[idx:(idx+1)]
                    select_T = source_camera.T[idx:(idx+1)]
                    select_focal_length = source_camera.focal_length[idx:(idx+1)]
                    select_source_camera = PerspectiveCameras(
                        focal_length=select_focal_length, 
                        R=select_R, 
                        T=select_T, 
                        device=select_R.device
                    )

                    test_source_nerf_out, test_source_metrics = model(
                        None,
                        select_source_camera,
                        select_source_image,
                        select_source_depth,
                        select_source_sem_label,
                        source_camera, 
                        source_image,
                        source_depth,
                    )
                    source_nerf_outs.append(test_source_nerf_out)
            
            else: # use the only source view
                select_source_image = source_image
                select_source_sem_label = source_sem_label
                select_source_camera = source_camera
                select_source_depth = source_depth
            
                test_source_nerf_out, test_source_metrics = model(
                    None,
                    select_source_camera,
                    select_source_image,
                    select_source_depth,
                    select_source_sem_label,
                    source_camera, 
                    source_image,
                    source_depth,
                )
                source_nerf_outs.append(test_source_nerf_out)

        if cfg.test.mode == "evaluation":
            # Update stats with the validation metrics.
            stats.update(test_metrics, stat_set="test")

        elif cfg.test.mode == "export_imgs":
            stats.update(test_metrics, stat_set="test")
            stats.print(stat_set="test")
            save_nerf_outputs(export_dir, batch_idx, test_nerf_out, source_nerf_outs, metadata)

        # skip the loop if there are enough examples
        if batch_idx > cfg.test.num_samples and cfg.test.num_samples != -1:
            break


@hydra.main(config_path=CONFIG_DIR, config_name="viewseg_replica_finetune")
def main(cfg: DictConfig):
    # Set the relevant seeds for reproducibility.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the testing is unlikely to finish in reasonable time."
        )
        device = "cpu"

    # load dataset
    train_dataset, val_dataset, test_dataset = get_viewseg_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=cfg.data.image_size,
        num_views=cfg.train.num_views,
        load_depth=cfg.test.use_depth,
    )
    print("data split: {}".format(cfg.test.split))
    if cfg.test.split == 'train':
        test_dataset = train_dataset
    elif cfg.test.split == 'val':
        test_dataset = val_dataset
    elif cfg.test.split == 'test':
        pass
    else:
        raise NotImplementedError("cannot recognize data split {}".format(cfg.test.split))

    # Init the test dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Initialize the Radiance Field model.
    if cfg.implicit_function.use_image_feats:
        scene_encoder = build_spatial_encoder(
            backbone=cfg.encoder.backbone,
            num_layers=cfg.encoder.num_layers,
            pretrained=False, # unnecessary to load ade20k weights in test mode
            norm_type=cfg.encoder.norm_type,
            use_first_pool=cfg.encoder.use_first_pool,
            index_interp=cfg.encoder.index_interp,
            index_padding=cfg.encoder.index_padding,
            upsample_interp=cfg.encoder.upsample_interp,
            feature_scale=cfg.encoder.feature_scale,
            model_path=cfg.encoder.pretrained_model_path,
        )
    else:
        scene_encoder = None

    model = SemanticRadianceFieldRenderer(
        image_size=cfg.data.render_size,
        n_pts_per_ray=cfg.raysampler.n_pts_per_ray,
        n_pts_per_ray_fine=cfg.raysampler.n_pts_per_ray,
        n_rays_per_image=cfg.raysampler.n_rays_per_image,
        min_depth=cfg.raysampler.min_depth,
        max_depth=cfg.raysampler.max_depth,
        stratified=cfg.raysampler.stratified,
        stratified_test=cfg.raysampler.stratified_test,
        chunk_size_test=cfg.raysampler.chunk_size_test,
        n_harmonic_functions_xyz=cfg.implicit_function.n_harmonic_functions_xyz,
        n_harmonic_functions_dir=cfg.implicit_function.n_harmonic_functions_dir,
        n_hidden_neurons_xyz=cfg.implicit_function.n_hidden_neurons_xyz,
        n_hidden_neurons_dir=cfg.implicit_function.n_hidden_neurons_dir,
        n_layers_xyz=cfg.implicit_function.n_layers_xyz,
        n_classes=cfg.implicit_function.n_classes,
        density_noise_std=cfg.implicit_function.density_noise_std,
        scene_encoder=scene_encoder,
        transform_to_source_view=cfg.implicit_function.transform_to_source_view,
        use_image_feats=cfg.implicit_function.use_image_feats,
        resnetfc=cfg.implicit_function.resnetfc,
        ignore_index=cfg.optimizer.ignore_index,
        use_depth=cfg.test.use_depth,
        use_view_dirs=cfg.implicit_function.use_view_dirs,
    )

    # Move the model to the relevant device.
    model.to(device)

    # Resume from the checkpoint.
    output_dir = os.path.join(hydra.utils.get_original_cwd(), 'checkpoints', cfg.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if cfg.test.epoch == 'None':
        checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.pth'.format(cfg.test.epoch))
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Model checkpoint {checkpoint_path} does not exist!")

    print(f"Loading checkpoint {checkpoint_path}.")
    loaded_data = torch.load(checkpoint_path)
    # Do not load the cached xy grid.
    # - this allows setting an arbitrary evaluation image size.
    state_dict = {
        k: v
        for k, v in loaded_data["model"].items()
        if "_grid_raysampler._xy_grid" not in k
    }
    state_dict = single_gpu_prepare(state_dict)

    stats = pickle.loads(loaded_data["stats"])
    print(f"   => resuming from epoch {stats.epoch}.")
    model.load_state_dict(state_dict, strict=False)

    # test_nerf writes visualization to export_dir
    if cfg.test.mode == "export_imgs":
        export_dir = os.path.join(output_dir, '{:0>4}_imgs_on_{}'.format(stats.epoch, cfg.data.dataset_name))
    elif cfg.test.mode == "export_video":
        export_dir = os.path.splitext(checkpoint_path)[0] + "_video"
    elif cfg.test.mode == 'export_mesh':
        export_dir = os.path.splitext(checkpoint_path)[0] + "_mesh"
    else:
        raise ValueError(f"Unknown test mode {cfg.test_mode}.")
    os.makedirs(export_dir, exist_ok=True)

    # prepare metadata, colormap and uvmap for visualization
    dataset_name = 'hypersim_sem_seg_{}'.format(cfg.test.split)
    metadata = MetadataCatalog.get(dataset_name)
    colormap = torch.FloatTensor(np.array(metadata.stuff_colors)) / 255.0
    colormap = torch.cat((colormap, torch.FloatTensor([[0.0, 0.0, 0.0]])), dim=0)
    uvmap = colormap[:, None]
    uvmap = torch.FloatTensor(uvmap)

    # Set the model to the eval mode.
    model.eval()

    # reseed to make sure we always get the same batch despite different model architecture
    torch.manual_seed(cfg.seed)

    # inference
    if cfg.test.mode == "export_imgs":
        export_imgs(cfg, test_dataloader, model, colormap, metadata, stats, export_dir, device)
    elif cfg.test.mode == "export_video":
        export_video(cfg, test_dataloader, model, colormap, export_dir, device)
    elif cfg.test.mode == "export_mesh":
        export_mesh(cfg, test_dataloader, model, colormap, export_dir, device)
    else:
        raise NotImplementedError("unknown test mode {}".format(cfg.test.mode))

    print("[done!]")
    

if __name__ == "__main__":
    main()
