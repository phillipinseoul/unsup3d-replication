import torch
import torch.nn as nn
import torchvision

EPS = 1e-7

class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()

        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 8, cout, kernel_size=1, stride=1, padding=0, bias=False)
        ]

        if activation is not None:
            network += [activation()]
        
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EDDeconv(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        
        # 1) downsampling network
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16 * 2, nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16 * 4, nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, zdim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        ]
 
        # 2) upsampling network
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)
        ]

        if activation is not None:
            network += [activation()]

        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        pass
    
    def normalize(self, x):
        pass

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        pass

    

