import jax
import jax.nn as nn
import jax.numpy as jnp

from nmax import Module
from cvax.modules import Dense, Conv2d, BatchNorm2d, MaxPool2d


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


class BottleneckBlock(Module):

    conv_proj: Module
    bn_proj: Module
    
    conv_in: Module
    bn_in: Module

    conv_bt: Module
    bn_bt: Module

    conv_out: Module
    bn_out: Module

    def __init__(self,
        rng,
        in_channels: int,
        bt_channels: int,
        out_channels: int,
        in_stride: int = 1,
        ):

        rngs = jax.random.split(rng, num=4)

        self.use_proj = (in_channels != out_channels or in_stride != 1)
        
        if self.use_proj:
            self.conv_proj = Conv2d(rngs[0], (out_channels, in_channels, 1, 1), stride=in_stride)
            self.bn_proj = BatchNorm2d(out_channels)

        self.conv_in = Conv2d(rngs[1], (bt_channels, in_channels, 1, 1), stride=in_stride)
        self.bn_in = BatchNorm2d(bt_channels)

        self.conv_bt = Conv2d(rngs[2], (bt_channels, bt_channels, 3, 3))
        self.bn_bt = BatchNorm2d(bt_channels)

        self.conv_out = Conv2d(rngs[3], (out_channels, bt_channels, 1, 1))
        self.bn_out = BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        if self.use_proj:
            residual = self.conv_proj(residual)
            residual = self.bn_proj(residual)
        
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = nn.relu(x)

        x = self.conv_bt(x)
        x = self.bn_bt(x)
        x = nn.relu(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        x = nn.relu(x + residual)

        return x