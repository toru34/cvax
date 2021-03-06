import jax
import jax.nn as nn

from nmax import Module
from cvax.modules import Conv2d, MaxPool2d, BatchNorm2d


class ResStem(Module):

    conv: Module
    bn: Module

    def __init__(self,
        rng,
        kernel_shape: tuple[int, int, int, int],
        stride: int,
        ):
        self.conv = Conv2d(rng, kernel_shape)
        self.bn = BatchNorm2d(kernel_shape[0])
        self.pool = MaxPool2d(kernel_width=3, stride=2, padding='SAME')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)
        x = self.pool(x)
        return x


class ResBlock(Module):
    def __init__(self,
        rng,
        kernel_shape_tuple: tuple[tuple[int, int, int, int], ...],
        ):

        for i, kernel_shape in enumerate(kernel_shape_tuple):
            self.add_module(f'conv{i}', Conv2d(rng, kernel_shape))
        
    def forward(self, x):
        # TODO:
        for module in self.modules:
            x = getattr(self, module)(x)
        
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
                (256, 64, 1, 1))),
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
                (512, 128, 1, 1))),
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
                (1024, 256, 1, 1))),
            ResBlock(rng, (
                (256, 1024, 1, 1),
                (256, 256, 3, 3),
                (1024, 256, 1, 1))))
        
        self.stage4 = ResStage(
            ResBlock(rng, (
                (512, 1024, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1))),
            ResBlock(rng, (
                (512, 2048, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1))),
            ResBlock(rng, (
                (512, 2048, 1, 1),
                (512, 512, 3, 3),
                (2048, 512, 1, 1))))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        return x