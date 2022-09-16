#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import os, sys
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import pickle
import itertools
import torch
import hydra
import submitit
from accelerate import Accelerator, DistributedDataParallelKwargs
from detectron2.data import MetadataCatalog
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.comm import get_rank

from viewseg.dataset import collate_fn, get_viewseg_datasets
from viewseg.utils import single_gpu_prepare

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
# compatible with old training history 
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
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device    
    print("Device", accelerator.device)

    # Resume from the checkpoint.
    output_dir = os.path.join(hydra.utils.get_original_cwd(), 'checkpoints', cfg.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if cfg.test.epoch == 'None':
        checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.pth'.format(cfg.test.epoch))
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Model checkpoint {checkpoint_path} does not exist!")
    loaded_data = torch.load(checkpoint_path)
    # Do not load the cached xy grid.
    # - this allows setting an arbitrary evaluation image size.
    state_dict = loaded_data["model"]
    state_dict = single_gpu_prepare(state_dict)
    stats = pickle.loads(loaded_data["stats"])
    print(f"   => resuming from epoch {stats.epoch}.")

    print("[detectron2] rank {}".format(get_rank()))

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
        raise NotImplementedError

    # Init the test dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    dataset_name = '{}_sem_seg_{}'.format(cfg.data.dataset_name, cfg.test.split)
    metadata = MetadataCatalog.get(dataset_name)
    result_dir = os.path.join(output_dir, '{:0>4}_eval_on_{}'.format(stats.epoch, cfg.data.dataset_name))
    print(result_dir)

    evaluator = SemSegEvaluator(dataset_name, output_dir=result_dir)
    evaluator.reset()

    pth_list = os.listdir(result_dir)
    conf_matrix_list = []
    predictions = []
    depth_metrics = {
        'absolute': [],
        'absrel': [],
        'thres 1.25': [],
        'thres 1.25^2': [],
        'thres 1.25^3': [],
    }
    print("loading results...")
    for result_fname in tqdm(pth_list):
        if not result_fname.startswith('results_'):
            continue 
        f = open(os.path.join(result_dir, result_fname), 'rb')
        gpu_results = pickle.load(f)
        conf_matrix_list.append(gpu_results['conf_matrix'])
        predictions.append(gpu_results['predictions'])
        gpu_depth_metrics = gpu_results['depth_metrics']

    print("aggregating...")
    evaluator._predictions = list(itertools.chain(*predictions))
    conf_matrix = np.zeros_like(conf_matrix_list[0])
    for gpu_conf_matrix in conf_matrix_list:
        conf_matrix += gpu_conf_matrix
    evaluator._conf_matrix = conf_matrix

    print("[detectron2 evaluation]")
    results = evaluator.evaluate()

    print("treating objects as a single class")
    print("[object and stuff evaluation]")
    # build a new conf matrix
    #          stuff   object
    #        -----------------
    # stuff  |       |       |
    #        | ------|-------|
    # object |       |       |
    #        -----------------
    conf_matrix_os = np.zeros_like(conf_matrix[:-1, :-1])
    stuff_list = [0, 1, 21]
    for gt_idx in range(conf_matrix_os.shape[0]):
        for pred_idx in range(conf_matrix_os.shape[1]):
            if gt_idx in stuff_list and pred_idx in stuff_list:  # stuff tp
                conf_matrix_os[0][0] += conf_matrix[gt_idx][pred_idx]
            elif gt_idx in stuff_list and pred_idx not in stuff_list:
                conf_matrix_os[0][1] += conf_matrix[gt_idx][pred_idx]
            elif gt_idx not in stuff_list and pred_idx not in stuff_list:
                conf_matrix_os[1][1] += conf_matrix[gt_idx][pred_idx]
            elif gt_idx not in stuff_list and pred_idx in stuff_list:
                conf_matrix_os[1][0] += conf_matrix[gt_idx][pred_idx]
            else:
                raise ValueError("should not reach this point")

    evaluator._conf_matrix[:-1, :-1] = conf_matrix_os
    results = evaluator.evaluate()
    print(results)

    print("[depth metrics]")
    
    depth_metrics = {
        'all': {
            'absolute': {'cnt': 0, 'total': 0},
            'absrel': {'cnt': 0, 'total': 0},
            'thres 1.25': {'cnt': 0, 'total': 0},
            'thres 1.25^2': {'cnt': 0, 'total': 0},
            'thres 1.25^3': {'cnt': 0, 'total': 0},
        },
        'stuff': {
            'absolute': {'cnt': 0, 'total': 0},
            'absrel': {'cnt': 0, 'total': 0},
            'thres 1.25': {'cnt': 0, 'total': 0},
            'thres 1.25^2': {'cnt': 0, 'total': 0},
            'thres 1.25^3': {'cnt': 0, 'total': 0},
        },
        'object': {
            'absolute': {'cnt': 0, 'total': 0},
            'absrel': {'cnt': 0, 'total': 0},
            'thres 1.25': {'cnt': 0, 'total': 0},
            'thres 1.25^2': {'cnt': 0, 'total': 0},
            'thres 1.25^3': {'cnt': 0, 'total': 0},
        },
    }

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

        # fetch depth pred
        depth_save_path = os.path.join(result_dir, 'predictions', '{}_depth.npz'.format(pair_ids[0]))
        depth_pred = np.load(depth_save_path)['depth']
        depth_pred = torch.FloatTensor(depth_pred)

        value_mask = torch.logical_not(torch.isnan(depth))
        value_mask = torch.logical_and(value_mask, depth > 0.1)
        stuff_mask = (sem_label == 0) # wall
        stuff_mask = torch.logical_or(stuff_mask, sem_label == 1) # floor
        stuff_mask = torch.logical_or(stuff_mask, sem_label == 21) # ceiling
        object_mask = torch.logical_not(stuff_mask)

        for sem_type in depth_metrics:
            if sem_type == 'all':
                valid_mask = value_mask
            elif sem_type == 'stuff':
                valid_mask = torch.logical_and(value_mask, stuff_mask)
            elif sem_type == 'object':
                valid_mask = torch.logical_and(value_mask, object_mask)
            
            thres_all = valid_mask.sum().item()
            sub_depth_pred = depth_pred[valid_mask]
            sub_depth = depth[valid_mask]

            l1_abs = torch.abs(sub_depth_pred - sub_depth)
            depth_metrics[sem_type]['absolute']['cnt'] += thres_all
            depth_metrics[sem_type]['absolute']['total'] += l1_abs.sum().item()

            absrel = torch.abs(sub_depth_pred - sub_depth) / sub_depth
            depth_metrics[sem_type]['absrel']['cnt'] += thres_all
            depth_metrics[sem_type]['absrel']['total'] += absrel.sum().item()

            thres = torch.max(
                sub_depth_pred / sub_depth, 
                sub_depth / sub_depth_pred
            )
            delta_1 = thres < 1.25
            depth_metrics[sem_type]['thres 1.25']['cnt'] += thres_all
            depth_metrics[sem_type]['thres 1.25']['total'] += delta_1.sum().item()

            delta_2 = thres < 1.25 ** 2
            depth_metrics[sem_type]['thres 1.25^2']['cnt'] += thres_all
            depth_metrics[sem_type]['thres 1.25^2']['total'] += delta_2.sum().item()

            delta_3 = thres < 1.25 ** 3
            depth_metrics[sem_type]['thres 1.25^3']['cnt'] += thres_all
            depth_metrics[sem_type]['thres 1.25^3']['total'] += delta_3.sum().item()


    if cfg.test.use_depth:
        print("[depth]")
        for sem_type in depth_metrics:
            print("Semantic: {}".format(sem_type))
            for metric in depth_metrics[sem_type]:
                depth_metrics[sem_type][metric]['result'] = depth_metrics[sem_type][metric]['total'] / depth_metrics[sem_type][metric]['cnt']
                print("\t{}: {}".format(metric, depth_metrics[sem_type][metric]['result']))


    print("[done]")



if __name__ == "__main__":
    main()
