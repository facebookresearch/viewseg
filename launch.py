# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import argparse
from typing import List
import hydra
from pathlib import Path


def run_experiment(
    experiment: List[str], 
    config_file: str, 
    config_path: str = None,
    local: bool = False, 
    mode: str = "train",
    model: str = "viewseg",
    nodes: int = 1,
    gpus: int = 8,
    debug: bool = False,
    largemem: bool = False,
):
    cmd = "python {}_{}.py".format(mode, model)

    # Multirun for slurm 
    if not local:
        cmd += " --multirun"
    
    cmd += " --config-name {}".format(config_file)

    if config_path is not None:
        cmd += " --config-path {}".format(config_path)
    
    # Modify some of the default settings
    if len(experiment) > 0:
        for setting in experiment:
            cmd += " {}".format(setting)

    # Add hydra launcher and other settings
    if local:
        cmd += " hydra/launcher=basic"
    else:
        cmd += " hydra/launcher=submitit_slurm"
        cmd += get_hydra_slurm_settings(
            partition="devlab" if debug else "learnlab",
            nodes=nodes,
            gpus=gpus,
            constraint="volta32gb" if largemem else None,
        )
    
    print(cmd)
    os.system(cmd)

def get_hydra_slurm_settings(
    partition="learnlab", 
    nodes=1,
    gpus=8,
    constraint=None):
    """
    Settings for launching on the cluster
    """
    cmd = ""
    cmd += " hydra.launcher.timeout_min=4320"
    cmd += " hydra.launcher.cpus_per_task=2"
    cmd += " hydra.launcher.gpus_per_node={}".format(gpus)
    cmd += " hydra.launcher.tasks_per_node={}".format(gpus)
    cmd += " hydra.launcher.mem_per_cpu=12000"
    cmd += " hydra.launcher.nodes={}".format(nodes)
    # use learnlab to avoid QOSMaxGRESPerUser
    cmd += " hydra.launcher.partition={}".format(partition)
    if constraint is not None:
        cmd += " hydra.launcher.constraint={}".format(constraint)
    return cmd

##############
# Entry Point
##############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --local is by default False
    parser.add_argument('--local', help='run locally', action='store_true', dest='local')
    parser.add_argument('--debug', help='visualize after 1 epoch and quit after 2 epochs for fast debugging', action='store_true', dest='debug')
    parser.add_argument('--largemem', help='request 32gb v100', action='store_true', dest='largemem')
    parser.add_argument('--mode', help='train or eval or test', type=str, default="train", dest='mode')
    parser.add_argument('--model', help='nerf or segcloud', type=str, default="viewseg", dest='model')
    parser.add_argument('--nodes', help='number of nodes', type=int, default=1, dest='nodes')
    parser.add_argument('--gpus', help='number of gpus per node', type=int, default=8, dest='gpus')
    parser.add_argument('--config', help='config file name', type=str, default="replica_sem_v1", dest='config')
    parser.add_argument('--checkpoint', help='checkpoint file full path', type=str, default=None, dest='checkpoint')
    parser.add_argument('--name', help='name of the run', type=str, default=None, dest='name')
    args = parser.parse_args()
    
    settings = []
    if args.name is not None:
        settings.append("hydra.launcher.name={} name={}".format(args.name, args.name))
    
    # Debug setting, use devlab partition, visualize after 1 epoch and quit after 2 epochs
    if args.debug:
       settings.append("validation_epoch_interval=1 optimizer.max_epochs=2")
    
    # Load from checkpoint with config file
    config_file = args.config
    config_path = None
    if args.checkpoint is not None:
        # For Hydra need to escape all the "=" in the checkpoint path
        settings.append("checkpoint_path='{}'".format(args.checkpoint))

        # Check if a config file exists in the checkpoint dir and if so load that file
        # and pass it in as the config file
        config_checkpoint_dir = Path(args.checkpoint).parent
        config_checkpoint_name = "config"
        config_checkpoint_fullpath = os.path.join(config_checkpoint_dir, config_checkpoint_name) + ".yaml"
        if os.path.isfile(config_checkpoint_fullpath):
            print("Using config file from: %s" % config_checkpoint_fullpath)
            config_file = config_checkpoint_name
            config_path = config_checkpoint_dir 

    # Run the experiment
    run_experiment(
        settings, 
        config_file, 
        config_path, 
        args.local, 
        args.mode,
        args.model,
        args.nodes,
        args.gpus,
        args.debug, 
        args.largemem
    )
