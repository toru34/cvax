import jax
import jax.nn as nn
import jax.numpy as jnp

from nmax import Module
from cvax.modules import Dense, Conv2d, MaxPool2d, BatchNorm2d


class ResStem(Module):

    conv: Module
    bn: Module

    def __init__(self,
        rng,
        kernel_shape: tuple[int, int, int, int],
        stride: int,
        ):
        self.conv = Conv2d(rng, kernel_shape, stride=stride)
        self.bn = BatchNorm2d(kernel_shape[0])
        self.pool = MaxPool2d(kernel_width=3, stride=2, padding='SAME')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)
        x = self.pool(x)
        return x


class ResHead(Module):

    dense: Module

    def __init__(self, rng, in_channels, out_dim):
        self.dense = Dense(rng, in_channels, out_dim)
    
    def forward(self, x):
        x = jnp.mean(x, axis=(2, 3))
        x = self.dense(x)
        return x


class ResBlock(Module):
    def __init__(self,
        rng,
        kernel_shape_tuple: tuple[tuple[int, int, int, int], ...],
        use_proj=False,
        halve_resolution=False,
        ):

        self.use_proj = use_proj
        self.halve_resolution = halve_resolution

        if self.use_proj or self.halve_resolution:
            in_channels = kernel_shape_tuple[0][1]
            out_channels = kernel_shape_tuple[-1][0]
            stride = 2 if self.halve_resolution else 1
            
            self.conv_proj = Conv2d(rng, (out_channels, in_channels, 1, 1), stride=stride)
            self.bn_proj = BatchNorm2d(out_channels)
        
        for i, kernel_shape in enumerate(kernel_shape_tuple):
            stride = 2 if i == 0 and self.halve_resolution else 1
            self.add_module(f'conv{i}', Conv2d(rng, kernel_shape, stride=stride))
            self.add_module(f'bn{i}', BatchNorm2d(kernel_shape[0]))
        
        self.n_convs = len(kernel_shape_tuple)
        
    def forward(self, x):
        residual = x
        if self.use_proj or self.halve_resolution:
            residual = self.conv_proj(residual)
            residual = self.bn_proj(residual)
        
        for i in range(self.n_convs):
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'bn{i}')(x)

            if i == self.n_convs - 1:
                x += residual
            
            x = nn.relu(x)
        
        return x


class ResStage(Module):
    def __init__(self,
        *resblock_tuple,
        ):
        self.resblock_tuple = tuple(resblock_tuple)
    
    def forward(self, x):
        for resblock in self.resblock_tuple:
            x = resblock(x)
        return x


class ResNet50(Module):

    stem: Module

    def __init__(self,
        rng
        ):
        self.stem = ResStem(rng, (64, 3, 7, 7), stride=2) # TODO: Change rng

        self.stage1 = ResStage(
            ResBlock(rng, (
                (64, 64, 1, 1),
                (64, 64, 3, 3),
                (256, 64, 1, 1)), use_proj=True),
            ResBlock(rng, (
                (64, 256, 1, 1),
                (64, 64, 3, 3),
                (256, 64, 1, 1))),
            ResBlock(rng, (
                (64, 256, 1, 1),
                (64, 64, 3, 3),
                (256, 64, 1, 1))))

        self.stage2 = ResStage(
            ResBlock(rng, (
                (128, 256, 1, 1),
                (128, 128, 3, 3),
                (512, 128, 1, 1)), halve_resolution=True),
            ResBlock(rng, (
                (128, 512, 1, 1),
                (128, 128, 3, 3),
                (512, 128, 1, 1))),
            ResBlock(rng, (
                (128, 512, 1, 1),
                (128, 128, 3, 3),
                (512, 128, 1, 1))),
            ResBlock(rng, (
                (128, 512, 1, 1),
                (128, 128, 3, 3),
                (512, 128, 1, 1))))
        
        self.stage3 = ResStage(
            ResBlock(rng, (
                (256, 512, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1)), halve_resolution=True),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))))
        
        self.stage4 = ResStage(
            ResBlock(rng, (
                (512, 1024, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1)), halve_resolution=True),
            ResBlock(rng, (
                (512, 2048, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1))),
            ResBlock(rng, (
                (512, 2048, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1))))
        
        self.head = ResHead(rng, 2048, 1000)

    def forward(self, x):
        x = self.stem(x) # (N, 64, 112, 112)
        x = self.stage1(x) # (N, 256, 56, 56)
        x = self.stage2(x) # (N, 512, 28, 28)
        x = self.stage3(x) # (N, 1024, 14, 14)
        x = self.stage4(x) # (N, 2048, 7, 7)
        x = self.head(x) # (N, 1000)
        return x