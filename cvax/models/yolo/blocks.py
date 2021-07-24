import jax.nn as nn

from nmax import Module
from cvax.modules import Conv2d, BatchNorm2d


class YOLOConvBlock(Module):

    conv: Module
    bn: Module

    def __init__(self,
        key,
        kernel_shape: tuple[int, int, int, int],
        stride: int = 1,
        batch_norm: bool = True,
        activation: bool = True,
        ):

        self.conv = Conv2d(key, kernel_shape=kernel_shape, stride=stride)


        if batch_norm:
            self.bn = BatchNorm2d(n_channels=kernel_shape[0])
        else:
            self.bn = None
        
        self.activation = activation
        
    def forward(self, x):

        x = self.conv(x)

        if self.bn:
            x = self.bn(x)
        
        if self.activation:
            x = nn.leaky_relu(x, negative_slope=0.1) # TODO: add activation option

        return x


class DarknetResBlock(Module):

    conv_in: Module
    conv_out: Module

    def __init__(self,
        key,
        in_channels: int,
        ):

        keys = jax.random.split(key)

        self.conv_in = YOLOConvBlock(keys[0], kernel_shape=(in_channels, in_channels, 1, 1))
        self.conv_out = YOLOConvBlock(keys[1], kernel_shape=(in_channels, in_channels, 3, 3))

    def forward(self, x):

        residual = x

        x = self.conv_in(x)
        x = self.conv_out(x)

        x += residual

        return x