import jax
import jax.nn as nn

from nmax import Module
from cvax.utils import key_generator
from cvax.modules.conv import Conv2d, DepthwiseConv2d


class MobileBottleneckBlock(Module):

    conv_in: Module
    depconv: Module
    conv_out: Module

    def __init__(self,
        key,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        channel_multiplier: int = 6,
        use_residual: bool = False,
        ):

        g = key_generator(key)

        bt_channels = channel_multiplier * in_channels

        self.conv_in = Conv2d(next(g), kernel_shape=(bt_channels, in_channels, 1, 1), activation=nn.relu6)
        self.depconv = DepthwiseConv2d(next(g), kernel_shape=(bt_channels, 1, 3, 3), stride=stride, activation=nn.relu6)
        self.conv_out = Conv2d(next(g), kernel_shape=(out_channels, bt_channels, 1, 1), activation=nn.relu6)

        self.use_residual = use_residual
    
    def forward(self, x):

        if self.use_residual:
            residual = x

        x = self.conv_in(x)
        x = self.depconv(x)
        x = self.conv_out(x)

        if self.use_residual:
            x += residual
        
        return x