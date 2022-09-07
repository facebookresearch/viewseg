#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import collections
import os
import pickle
import warnings
import pdb
import hydra
import numpy as np
import logging
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch
from omegaconf import DictConfig, OmegaConf
from visdom import Visdom
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
import submitit
from PIL import Image
import logging
from detectron2.data import MetadataCatalog

from viewseg.dataset import collate_fn, get_viewseg_datasets
from viewseg.renderer import SemanticRadianceFieldRenderer
from viewseg.vis import visualize_nerf_outputs
from viewseg.nerf.stats import Stats
from viewseg.encoder import build_spatial_encoder
from viewseg.utils import single_gpu_prepare

torch.autograd.set_detect_anomaly(True)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


def train(
    cfg,
    train_dataloader,
    model,
    optimizer,
    accelerator,
    stats,
    visuals_cache
):
    for iteration, batch in enumerate(train_dataloader):
        # This should now be batched and on the correct device
        image = batch['target_image']
        sem_label = batch['target_sem_label']
        camera = batch['target_camera']
        depth = batch['target_depth']
        camera_idx = batch['target_camera_idx']
        source_image = batch['source_image']
        source_camera = batch['source_camera']
        source_sem_label = batch['source_sem_label']
        source_depth = batch['source_depth']

        # prepare results
        nerf_out = {}
        metrics = {}
        lbd_color = cfg.optimizer.lbd_color
        lbd_semantic = cfg.optimizer.lbd_semantic
        lbd_depth = cfg.optimizer.lbd_depth
        loss = 0.0

        # Run the forward pass of the model.
        
        # use target view
        optimizer.zero_grad()
        if cfg.train.use_target_view:
            nerf_out, target_metrics = model(
                camera_hash=camera_idx if cfg.data.precache_rays else None,
                camera=camera,
                image=image,
                depth=depth,
                sem_label=sem_label,
                source_camera=source_camera,
                source_image=source_image,        
            )
            metrics.update(target_metrics)

        # sometimes semantic_gt is filled with -1 and loss is nan
        # See https://github.com/pytorch/pytorch/issues/70348
        if torch.isnan(metrics['semantic_coarse']):
            metrics['semantic_coarse'].zero_()
        if torch.isnan(metrics['semantic_fine']):
            metrics['semantic_fine'].zero_()
        

        if cfg.train.use_target_view:
            loss += lbd_color * metrics["mse_coarse"] + lbd_color * metrics["mse_fine"]
            loss += lbd_semantic * metrics["semantic_coarse"] + lbd_semantic * metrics["semantic_fine"]
            if cfg.implicit_function.use_depth:
                loss += lbd_depth * metrics["depth_coarse"] + lbd_depth * metrics["depth_fine"]
            accelerator.backward(loss)
            optimizer.step()

        # use source view
        if cfg.train.use_source_view:
            optimizer.zero_grad()
            loss = 0.0
            if cfg.train.num_views > 1:
                # multiple source views, random select a source view
                idx = int(np.random.randint(cfg.train.num_views, size=1))
                select_source_image = source_image[:, idx]
                select_source_sem_label = source_sem_label[:, idx]
                if source_depth is None:
                    select_source_depth = None
                else:
                    select_source_depth = source_depth[:, idx]
                select_R = source_camera.R[idx:(idx+1)]
                select_T = source_camera.T[idx:(idx+1)]
                select_focal_length = source_camera.focal_length[idx:(idx+1)]
                select_source_camera = PerspectiveCameras(focal_length=select_focal_length, R=select_R, T=select_T, device=select_R.device)
            else: # use the only source view
                select_source_image = source_image
                select_source_sem_label = source_sem_label
                if source_depth is None:
                    select_source_depth = None
                else:
                    select_source_depth = source_depth[:, idx]
                select_source_camera = source_camera
                
            source_nerf_out, source_metrics = model(
                None,
                select_source_camera,
                select_source_image,
                select_source_depth,
                select_source_sem_label,
                source_camera,
                source_image,
            )
            source_metrics = {"source_%s" % k: v for k, v in source_metrics.items()}
            metrics.update(source_metrics)

        if cfg.train.use_source_view:
            loss += lbd_color * metrics["source_mse_coarse"] + lbd_color * metrics["source_mse_fine"]
            loss += lbd_semantic * metrics["source_semantic_coarse"] + lbd_semantic * metrics["source_semantic_fine"]
            if cfg.implicit_function.use_depth:
                loss += lbd_depth * metrics["source_depth_coarse"] + lbd_depth * metrics["source_depth_fine"]

            # Take the training step using accelerator
            accelerator.backward(loss)
            optimizer.step()

        # Update stats with the current metrics.
        stats.update(
            {"loss": float(loss), **metrics},
            stat_set="train",
        )

        if accelerator.is_local_main_process and iteration % cfg.stats_print_interval == 0:
            stats.print(stat_set="train")

        # Update the visualization cache.
        visuals_cache.append(
            {
                "camera": camera.cpu(),
                "camera_idx": camera_idx,
                "image": image.cpu().detach(),
                "source_image": source_image.cpu().detach(),
                "source_rgb_fine": source_nerf_out["rgb_fine"].cpu().detach() if cfg.train.use_source_view else None,
                "source_rgb_coarse": source_nerf_out["rgb_coarse"].cpu().detach() if cfg.train.use_source_view else None,
                "source_rgb_gt": source_nerf_out["rgb_gt"].cpu().detach() if cfg.train.use_source_view else None,
                "rgb_fine": nerf_out["rgb_fine"].cpu().detach(),
                "rgb_coarse": nerf_out["rgb_coarse"].cpu().detach(),
                "rgb_gt": nerf_out["rgb_gt"].cpu().detach(),
                "source_semantic_fine": source_nerf_out["semantic_fine"].cpu().detach() if cfg.train.use_source_view else None,
                "source_semantic_coarse": source_nerf_out["semantic_coarse"].cpu().detach() if cfg.train.use_source_view else None,
                "source_semantic_gt": source_nerf_out["semantic_gt"].cpu().detach() if cfg.train.use_source_view else None,
                "semantic_fine": nerf_out["semantic_fine"].cpu().detach(),
                "semantic_coarse": nerf_out["semantic_coarse"].cpu().detach(),
                "semantic_gt": nerf_out["semantic_gt"].cpu().detach(),
                "coarse_ray_bundle": nerf_out["coarse_ray_bundle"],
            }
        )

        if cfg.train.num_samples != -1 and iteration > cfg.train.num_samples:
            break


@hydra.main(config_path=CONFIG_DIR, config_name="replica")
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
        #job_id = job_env.job_id
    except RuntimeError:
        print("Running locally")
        #job_id = ""
        
    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = os.path.join(hydra.utils.get_original_cwd(), 'checkpoints', cfg.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('root')
    logger.critical("launching experiment {}".format(cfg.experiment_name))

    # Set up the accelerator for multigpu training
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device    
    logger.info("Device {}".format(accelerator.device))

    # Pretty print the config
    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))

    # pass config check
    if cfg.validation_epoch_interval == 0 or cfg.checkpoint_epoch_interval == 0:
        raise ValueError("epoch_interval cannot be 0.")
        
    # Device on which to run.
    if device == "cpu":
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )

    # Initialize the Radiance Field model.
    if cfg.implicit_function.use_image_feats:
        scene_encoder = build_spatial_encoder(
            backbone=cfg.encoder.backbone,
            bn=cfg.encoder.bn,
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
        #image_size=cfg.data.image_size,
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
        use_depth=cfg.implicit_function.use_depth,
        use_view_dirs=cfg.implicit_function.use_view_dirs,
    )

    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0
    

    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    if len(cfg.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.resume and os.path.isfile(checkpoint_path):
            logger.info(f"Resuming from checkpoint {checkpoint_path}.")
            if not accelerator.is_local_main_process:
                map_location = {'cuda:0': 'cuda:%d' % accelerator.local_process_index}
            else:
                # Running locally
                map_location = "cuda:0"
            
            loaded_data = torch.load(checkpoint_path, map_location=map_location)

            state_dict = loaded_data["model"]

            # Single GPU vs MultiGPU
            if accelerator.distributed_type == DistributedType.NO:
                logger.info("Single gpu training.")
                state_dict = single_gpu_prepare(state_dict)

            model.load_state_dict(state_dict, strict=False)

            # finetune on replica: do not load optimizer and stats
            # continue training: load optimizer and stats
            if not cfg.data.dataset_name.startswith('replica'):
                stats = pickle.loads(loaded_data["stats"])
                logger.info(f"   => resuming from epoch {stats.epoch}.")
                optimizer_state_dict = loaded_data["optimizer"]
                start_epoch = stats.epoch + 1
        else:
            logger.info("Start from scratch.")
        
        # Check if a config file exists in the checkpoint dir otherwise save it
        # for use during testing/evaluation
        if not os.path.isfile(os.path.join(checkpoint_dir, "config.yaml")):
            OmegaConf.save(config=cfg, f=os.path.join(checkpoint_dir, "config.yaml"))

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine",
            "source_mse_coarse", "source_mse_fine", "source_psnr_coarse", "source_psnr_fine",
            "semantic_coarse", "semantic_fine",
            "source_semantic_coarse", "source_semantic_fine",
            "depth_coarse", "depth_fine",
            "source_depth_coarse", "source_depth_fine",
            "sec/it"],
        )

    # Learning rate scheduler setup.

    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # Initialize the cache for storing variables needed for visualization.
    visuals_cache = collections.deque(maxlen=cfg.visualization.history_size)

    # Init the visualization visdom env.
    if accelerator.is_main_process and cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
            env=cfg.experiment_name,
        )
    else:
        viz = None

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_viewseg_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=cfg.data.image_size,
        num_views=cfg.train.num_views,
        load_depth=cfg.implicit_function.use_depth,
    )

    logger.info("train has {} examples".format(len(train_dataset)))
    logger.info("val has {} examples".format(len(val_dataset)))

    dataset_name = 'hypersim_sem_seg_{}'.format('train')
    metadata = MetadataCatalog.get(dataset_name)

    if cfg.data.precache_rays:
        # Precache the projection rays.
        model.eval()
        with torch.no_grad():
            for dataset in (train_dataset, val_dataset):
                cache_cameras = [e["camera"].to(device) for e in dataset]
                cache_camera_hashes = [e["camera_idx"] for e in dataset]
                model.precache_rays(cache_cameras, cache_camera_hashes)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
        sampler=torch.utils.data.RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=cfg.optimizer.max_epochs,
        ),
    )

    # Prepare the model for accelerate and move to the relevant device
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Set the model to the training mode.
    model.train()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        stats.new_epoch()  # Init a new epoch.
        train(cfg, train_dataloader, model, optimizer, accelerator, stats, visuals_cache)

        # Adjust the learning rate.
        lr_scheduler.step()

        # Validation
        if epoch % cfg.validation_epoch_interval == 0:
            # Sample a validation camera/image.
            val_batch = next(val_dataloader.__iter__())
            # This should now be on the correct device
            val_image = val_batch['target_image']
            val_sem_label = val_batch['target_sem_label']
            val_camera = val_batch['target_camera']
            val_camera_idx = val_batch['target_camera_idx']
            val_depth = val_batch['target_depth']

            val_source_image = val_batch['source_image']
            val_source_camera = val_batch['source_camera']
            val_source_sem_label = val_batch['source_sem_label']
            val_source_depth = val_batch['source_depth']

            # Activate eval mode of the model (lets us do a full rendering pass).
            model.eval()
            with torch.no_grad():
                val_nerf_out, val_metrics = model(
                    val_camera_idx if cfg.data.precache_rays else None,
                    val_camera,
                    val_image,
                    val_depth,
                    val_sem_label,
                    val_source_camera, 
                    val_source_image
                )

                # sometimes semantic_gt is filled with -1 and loss is nan
                # See https://github.com/pytorch/pytorch/issues/70348
                if torch.isnan(val_metrics['semantic_coarse']):
                    val_metrics['semantic_coarse'].zero_()
                if torch.isnan(val_metrics['semantic_fine']):
                    val_metrics['semantic_fine'].zero_()

                # rerun on val_image
                if cfg.train.use_source_view:
                    if cfg.train.num_views > 1:
                        # multiple source views, random select a source view
                        idx = int(np.random.randint(cfg.train.num_views, size=1))
                        select_val_source_image = val_source_image[:, idx]
                        select_val_source_sem_label = val_source_sem_label[:, idx]
                        if val_source_depth is None:
                            select_val_source_depth = None
                        else:
                            select_val_source_depth = val_source_depth[:, idx]
                        select_R = val_source_camera.R[idx:(idx+1)]
                        select_T = val_source_camera.T[idx:(idx+1)]
                        select_focal_length = val_source_camera.focal_length[idx:(idx+1)]
                        select_val_source_camera = PerspectiveCameras(focal_length=select_focal_length, R=select_R, T=select_T, device=select_R.device)
                    else: # use the only source view
                        select_val_source_image = val_source_image
                        select_val_source_sem_label = val_source_sem_label
                        if val_source_depth is None:
                            select_val_source_depth = None
                        else:
                            select_val_source_depth = val_source_depth
                        select_val_source_camera = val_source_camera
                 
                    source_val_nerf_out, source_val_metrics = model(
                        val_camera_idx if cfg.data.precache_rays else None,
                        select_val_source_camera,
                        select_val_source_image,
                        select_val_source_depth,
                        select_val_source_sem_label,
                        val_source_camera, 
                        val_source_image
                    )

                    source_val_nerf_out = {"source_%s" % k: v for k, v in source_val_nerf_out.items()}
                    val_nerf_out.update(source_val_nerf_out)

                    source_val_metrics = {"source_%s" % k: v for k, v in source_val_metrics.items()}
                    val_metrics.update(source_val_metrics)


            # Update stats with the validation metrics.
            stats.update(val_metrics, stat_set="val")
            if accelerator.is_local_main_process:
                stats.print(stat_set="val")

            if accelerator.is_main_process and viz is not None:
                # Plot that loss curves into visdom.
                stats.plot_stats(
                    viz=viz,
                    visdom_env=cfg.experiment_name,
                    plot_file=None,
                )
                # Visualize the intermediate results.
                visualize_nerf_outputs(
                    val_nerf_out, visuals_cache, viz, cfg.experiment_name, metadata,
                )

            # Set the model back to train mode.
            model.train()

        # Checkpoint.
        accelerator.wait_for_everyone()
        if (
            accelerator.is_main_process
            and epoch % cfg.checkpoint_epoch_interval == 0
            and len(cfg.checkpoint_path) > 0
        ):
            logger.info(f"Storing checkpoint {checkpoint_path}.")
            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, checkpoint_path)

            epoch_checkpoint_path = checkpoint_path.replace('.pth', '_{}.pth'.format(epoch))
            logger.info(f"Storing checkpoint {epoch_checkpoint_path}.")
            torch.save(data_to_store, epoch_checkpoint_path)


if __name__ == "__main__":
    main()
