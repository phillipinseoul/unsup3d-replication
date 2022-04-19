import os
import math
import glob
import torch
import torch.nn as nn
import torchvision

import networks
import utils
from renderer import Renderer

# from . import networks      # import everything from networks.py
# from . import utils         # import everything from utils.py
# from .renderer import Renderer

# Define class Unsup3D()
class Unsup3D():
    def __init__(self, cfgs):
        # initialize parameters for Unsup3D
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.use_conf_map = cfgs.get('use_conf_map', True)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lam_depth_sm = cfgs.get('lam_depth_sm', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        
        # import neural renderer
        self.renderer = Renderer(cfgs)

        # define networks and optimizers
        self.netD = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)

        self.network_names = [k for k in vars(self) if 'net' in k]

        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
        )

        # define other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_params_names = ['PerceptualLoss']

        # define depth rescaler: -1 ~ 1 --> min_depth ~ max_depth
        self.depth_rescaler = lambda d : (1+d)/2 * self.max_depth + (1-d)/2 * self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 * self.max_amb_light + (1-x)/2 * self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 * self.max_diff_light + (1-x)/2 * self.min_diff_light
    
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







