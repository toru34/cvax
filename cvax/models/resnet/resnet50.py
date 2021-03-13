import jax
import jax.nn as nn
import jax.numpy as jnp

from nmax import Module, ModuleTuple
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


class ResNet50(Module):
    
    stem: Module

    stage1: Module
    stage2: Module
    stage3: Module
    stage4: Module

    head: Module

    def __init__(self,
        rng,
        out_dim: int = 1000,
        ):

        rngs = jax.random.split(rng, num=28)

        self.stem = ResStem(rngs[0], (64, 3, 7, 7), stride=2)

        self.stage1 = ModuleTuple((
            BottleneckBlock(rngs[1], 64, 64, 256),
            BottleneckBlock(rngs[2], 256, 64, 256),
            BottleneckBlock(rngs[3], 256, 64, 256),
        ))

        self.stage2 = ModuleTuple((
            BottleneckBlock(rngs[4], 256, 128, 512, in_stride=2),
            BottleneckBlock(rngs[5], 512, 128, 512),
            BottleneckBlock(rngs[6], 512, 128, 512),
            BottleneckBlock(rngs[7], 512, 128, 512),
        ))

        self.stage3 = ModuleTuple((
            BottleneckBlock(rngs[8], 512, 256, 1024, in_stride=2),
            BottleneckBlock(rngs[9], 1024, 256, 1024),
            BottleneckBlock(rngs[10], 1024, 256, 1024),
            BottleneckBlock(rngs[11], 1024, 256, 1024),
            BottleneckBlock(rngs[12], 1024, 256, 1024),
            BottleneckBlock(rngs[13], 1024, 256, 1024),
        ))

        self.stage4 = ModuleTuple((
            BottleneckBlock(rngs[14], 1024, 512, 2048, in_stride=2),
            BottleneckBlock(rngs[15], 2048, 512, 2048),
            BottleneckBlock(rngs[16], 2048, 512, 2048),
        ))

        self.head = ResHead(rngs[17], 2048, out_dim)
    
    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.head(x)

        return x