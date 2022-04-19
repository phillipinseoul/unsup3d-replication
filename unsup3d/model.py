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

        if self.use_conf_map:
            self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128)

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
        self.optimizer_names = []
        
        # init optimizer for each network in Unsup3D
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
    
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
        loss = (im1 - im2).abs()

        if conf_sigma is not None:
            loss = loss * 2 ** 0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, input):
        """Feedforward once."""

        # ground truth
        if self.load_gt_depth:
            input, depth_gt = input

        self.input_im = input.to(self.device) *2.-1.
        b, c, h, w = self.input_im.shape

        ### predict canonical depth (netD)
        self.canon_depth_raw = self.netD(self.input_im).squeeze(1)      # B x H x W
        self.canon_depth = self.canon_depth_raw - self.warp_canon_depth.view(b, -1).mean(1).view(b, 1, 1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ### optional depth smoothness loss (only used in synthetic car experiments)
        ''' Implement later '''
    
        ### clamp border depth
        depth_boarder = torch.zeros(1, h, w - 4).to(self.input_im.device)
        depth_boarder = nn.functional.pad(depth_boarder, (2, 2), mode='constant', value=1)
        self.canon_depth = self.canon_depth * (1 - depth_boarder) + depth_boarder * self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)   # flip

        ### predict canonical albedo (netA)
        self.canon_albedo = self.netA(self.input_im)
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)    # flip

        ### predict confidence map
        ''' Implement later '''

        ### predict lighting (netL)
        canon_light = self.netL(self.input_im).repeat(2, 1)     # B x 4
        self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])    # ambience term (주변광)
        self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term (분산광)
        
        canon_light_dxy = canon_light[:, 2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b * 2, 1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light ** 2).sum(1, keepdim=True)) ** 0.5     # diffuse light direction  

        ### shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        
        # use ambience lighting and diffuse lighting to compute the shading
        canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1, 1) * self.canon_diffuse_shading
        
        # create canonical image
        self.canon_im = (self.canon_albedo / 2 + 0.5) * canon_shading *2-1
                
        ### predict viewpoint transformation (netV)
        self.view = self.netV(self.input_im).repeat(2, 1)
        self.view = torch.cat([
            self.view[:, :3] * (math.pi/180) * self.xyz_rotation_range,
            self.view[:, 3:5] * self.xy_translation_range,
            self.view[:, 5:] * self.z_translation_range
        ], 1)

        ### reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)

        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        
        # compute the reconstruction using the canonical image & depth map
        self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (self.recon_depth < (self.max_depth + margin)).float()  # `invalid border pixels` have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[: b] * recon_im_mask[b: ]            # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2, 1, 1).unsqueeze(1).detach()
        
        # image reconstruction!
        self.recon_im = self.recon_im * recon_im_mask_both

        ### render symmetry axis
        canon_sym_axis = torch.zeros(h, w).to(self.input_im.device)
        canon_sym_axis[:, w//2 - 1 : w//2 + 1] = 1
        self.recon_sym_axis = nn.functional.grid_sample(canon_sym_axis.repeat(b * 2, 1, 1, 1), grid_2d_from_canon, mode='bilinear')
        self.recon_sym_axis = self.recon_sys_axis * recon_im_mask_both
        green = torch.FloatTensor([-1, 1, -1]).to(self.input_im.device).view(1, 3, 1, 1)
        self.input_im_symline = (0.5 * self.recon_sym_axis) * green + (1 - 0.5 * self.recon_sym_axis) * self.input_im_repeat(2, 1, 1, 1)

        ### loss function
        self.loss_l1_im = self.photometric_loss(self.recon_im[: b], self.input_im, mask=recon_im_mask_both[: b], conf_sigma=self.conf_sigma_l1)

        



    def visualize(self, logger, total_iter, max_bs=25):
        pass

    def save_results(self, save_dir):
        pass

    def save_scores(self, path):
        pass







