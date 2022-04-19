import os
import sys
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import zipfile


def setup_runtime(args):
    # set up CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device_id)
    if torch.cuda_is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda_is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load configurations (cfgs)
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    print(f'Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}')
    return cfgs


def load_yaml(path):
    print(f'loading configs from {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def dump_yaml(path, cfgs):
    print(f'saving configs to {path}')
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)

def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


