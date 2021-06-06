import jax.nn as nn

from nmax import YOLOConvBlock
from nmax.module import Module
from cvax.modules import Conv2d, BatchNorm2d

class YOLOConvBlock(Module):

    conv: Module
    bn: Module

    def __init__(self,
        key,
        kernel_shape: tuple[int, int, int, int],
        stride: int,
        ):

        self.conv = Conv2d(key, kernel_shape=kernel_shape, stride=stride)
        self.bn = BatchNorm2d(n_channels=kernel_shape[0])
    
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        return x