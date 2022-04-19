import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
import networks
import utils

# from . import networks      # import everything from networks.py
# from . import utils         # import everything from utils.py
# from .renderer import Renderer

# Define class Unsup3D()
class Unsup3D():
    def __init__(self, cfgs):
        # initialize parameters for Unsup3D

        # define networks and optimizers

        # define other parameters

        # define depth rescaler


        pass
    
    def init_optimizers(self):
        pass
    
    def load_model_state(self, cp):
        pass

    def load_optimizer_state(self, cp):
        pass

    def get_model_state(self):
        pass

    def get_optimizer_state(self):
        pass

    def to_device(self, device):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        pass

    def backward(self):
        pass

    def forward(self, input):
        pass

    def visualize(self, logger, total_iter, max_bs=25):
        pass

    def save_results(self, save_dir):
        pass

    def save_scores(self, path):
        pass







