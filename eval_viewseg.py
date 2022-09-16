#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import os, sys
import warnings
import numpy as np
from tqdm import tqdm
import collections
from omegaconf import DictConfig
from PIL import Image
import imageio
import pickle

import torch
import hydra
import submitit
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.io import save_obj

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.comm import get_rank
from viewseg.dataset import collate_fn, get_viewseg_datasets, DEFAULT_DATA_ROOT
from viewseg.renderer import SemanticRadianceFieldRenderer
from viewseg.vis import visualize_nerf_outputs, save_nerf_outputs
from viewseg.nerf.stats import Stats
from viewseg.encoder import build_spatial_encoder
from viewseg.utils import single_gpu_prepare


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
sys.modules['panonerf'] = sys.modules['viewseg']


@hydra.main(config_path=CONFIG_DIR, config_name="viewseg_replica_finetune")
def main(cfg: DictConfig):
    try:
        # Only needed when launching on cluster with slurm
        job_env = submitit.JobEnvironment()
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        hostname_first_node = (
            os.popen("scontrol show hostnames $SLURM_JOB_NODELIST").read().split("\n")[0]
        )
        print("[launcher] Using the following MASTER_ADDR: {}".format(hostname_first_node))
        os.environ["MASTER_ADDR"] = hostname_first_node
        os.environ["MASTER_PORT"] = "42918"
        job_id = job_env.job_id
    except RuntimeError:
        print("Running locally")
        job_id = ""

    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set up the accelerator for multigpu training
    print("initialize ddp scaler")
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device    
    print("Device", accelerator.device)

    # Initialize the Radiance Field model.
    if cfg.implicit_function.use_image_feats:
        scene_encoder = build_spatial_encoder(
            backbone=cfg.encoder.backbone,
            num_layers=cfg.encoder.num_layers,
            pretrained=cfg.encoder.pretrained,
            norm_type=cfg.encoder.norm_type,
            use_first_pool=cfg.encoder.use_first_pool,
            index_interp=cfg.encoder.index_interp,
            index_padding=cfg.encoder.index_padding,
            upsample_interp=cfg.encoder.upsample_interp,
            feature_scale=cfg.encoder.feature_scale,
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
        use_depth=cfg.test.use_depth, # enable depth in test time
        use_view_dirs=cfg.implicit_function.use_view_dirs,
    )

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
    if not accelerator.is_local_main_process:
        map_location = {'cuda:0': 'cuda:%d' % accelerator.local_process_index}
    else:
        # Running locally
        map_location = "cuda:0"

    loaded_data = torch.load(checkpoint_path, map_location=map_location)
    # Do not load the cached xy grid.
    # - this allows setting an arbitrary evaluation image size.
    state_dict = {
        k: v
        for k, v in loaded_data["model"].items()
        if "_grid_raysampler._xy_grid" not in k
    }
    state_dict = loaded_data["model"]
    state_dict = single_gpu_prepare(state_dict)
    model.load_state_dict(state_dict, strict=False)
    stats = pickle.loads(loaded_data["stats"])
    print(f"   => resuming from epoch {stats.epoch}.")

    # Load the test data.
    if cfg.test.mode == "evaluation":
        train_dataset, val_dataset, test_dataset = get_viewseg_datasets(
            dataset_name=cfg.data.dataset_name,
            image_size=cfg.data.image_size,
            num_views=cfg.train.num_views,
            load_depth=cfg.test.use_depth,
        )
        #test_dataset = train_dataset
    elif cfg.test.mode == "export_imgs":
        train_dataset, val_dataset, test_dataset = get_viewseg_datasets(
            dataset_name=cfg.data.dataset_name,
            image_size=cfg.data.image_size,
            num_views=cfg.train.num_views,
            load_depth=cfg.test.use_depth,
        )
        export_dir = os.path.splitext(checkpoint_path)[0] + '_' + cfg.test.split + "_imgs"
        os.makedirs(export_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown test mode {cfg.test_mode}.")

    print("data split: {}".format(cfg.test.split))
    if cfg.test.split == 'train':
        test_dataset = train_dataset
    elif cfg.test.split == 'val':
        test_dataset = val_dataset
    elif cfg.test.split == 'test':
        pass
    else:
        raise NotImplementedError

    # Init the test dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    if cfg.test.mode == "evaluation" or cfg.test.mode == "export_imgs":
        # Init the test stats object.
        eval_stats = ["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"]
    elif cfg.test.mode == "export_video":
        # Init the frame buffer.
        frame_paths = []

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # Set the model to the eval mode.
    model.eval()

    print("[detectron2] rank {}".format(get_rank()))

    if cfg.data.dataset_name.startswith('hypersim'):
        dataset_name = 'hypersim_sem_seg_val'
    elif cfg.data.dataset_name.startswith('replica'):
        dataset_name = 'replica_sem_seg_val'
    else:
        raise NotImplementedError("cannot find sem seg dataset for {} in detectron2".format(cfg.data.dataset_name))

    metadata = MetadataCatalog.get(dataset_name)
    result_dir = os.path.join(output_dir, '{:0>4}_eval_on_{}'.format(stats.epoch, cfg.data.dataset_name))
    os.makedirs(result_dir, exist_ok=True)
    evaluator = SemSegEvaluator(dataset_name, output_dir=result_dir)
    evaluator.reset()

    results = []

    print(next(model.parameters()).device)

    depth_metrics = {
        'absolute': [],
        'absrel': [],
        'thres 1.25': [],
        'thres 1.25^2': [],
        'thres 1.25^3': [],
    }

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
        pair_ids = test_batch['pair_id']

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

            # to speed up the full evaluation,
            # we do not run on source views
            source_nerf_outs = []

        if cfg.data.dataset_name == 'hypersim':
            dataset_seg_dir = os.path.join(DEFAULT_DATA_ROOT, 'hypersim_sem_seg', 'images')
        elif cfg.data.dataset_name == 'replica':
            dataset_seg_dir = os.path.join(DEFAULT_DATA_ROOT, 'replica_sem_seg', 'images')
        else:
            raise NotImplementedError("cannot find sem seg dataset for {} in detectron2".format(cfg.data.dataset_name))

        sem_pred = test_nerf_out['semantic_fine']
        sem_pred = sem_pred.permute(0, 3, 1, 2)[0]
        inputs = [{
            'file_name': os.path.join(dataset_seg_dir, cfg.test.split, '{}.jpg'.format(pair_ids[0]))
        }]
        predictions = [{
            'sem_seg': sem_pred,
        }]
        evaluator.process(inputs, predictions)

        if cfg.test.use_depth:
            # depth prediction
            depth_pred = test_nerf_out['depth_fine'][..., 0]

            # save depth instead
            depth_numpy = depth_pred.cpu().numpy()
            depth_save_dir = os.path.join(result_dir, 'predictions')
            os.makedirs(depth_save_dir, exist_ok=True)
            depth_save_path = os.path.join(depth_save_dir, '{}_depth.npz'.format(pair_ids[0]))
            np.savez_compressed(depth_save_path, depth=depth_numpy)

            valid_mask = torch.logical_not(torch.isnan(depth))
            thres_all = valid_mask.sum()
            depth_pred = depth_pred[valid_mask]
            depth = depth[valid_mask]

            depth_metrics['absolute'].append(torch.abs(depth_pred - depth).mean().item())
            depth_metrics['absrel'].append((torch.abs(depth_pred - depth) / depth).mean().item())
            thres = torch.max(depth_pred / depth, depth / depth_pred)
            depth_metrics['thres 1.25'].append(((thres < 1.25).sum() / thres_all).item())
            depth_metrics['thres 1.25^2'].append(((thres < 1.25 ** 2).sum() / thres_all).item())
            depth_metrics['thres 1.25^3'].append(((thres < 1.25 ** 3).sum() / thres_all).item())


        if cfg.test.mode == "evaluation":
            # Update stats with the validation metrics.
            stats.update(test_metrics, stat_set="test")

        elif cfg.test.mode == "export_imgs":
            stats.update(test_metrics, stat_set="test")
            stats.print(stat_set="test")
            save_nerf_outputs(export_dir, batch_idx, test_nerf_out, source_nerf_outs, metadata)

    print("saving results...")
    results = {
        'conf_matrix': evaluator._conf_matrix,
        'predictions': evaluator._predictions,
        'depth_metrics': depth_metrics if cfg.test.use_depth else None,
    }
    with open(os.path.join(result_dir, 'results_{}.pth'.format(get_rank())), "wb") as f:
        pickle.dump(results, f)

    if cfg.test.use_depth:
        for metric in depth_metrics:
            depth_metrics[metric] = np.array(depth_metrics[metric]).mean()

        print(depth_metrics)

    print("[done]")



if __name__ == "__main__":
    main()
