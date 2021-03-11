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

        self.use_proj = (in_channels != out_channels or in_stride != 1)
        
        if self.use_proj:
            self.conv_proj = Conv2d(rng, (out_channels, in_channels, 1, 1), stride=in_stride)
            self.bn_proj = BatchNorm2d(out_channels)

        self.conv_in = Conv2d(rng, (bt_channels, in_channels, 1, 1), stride=in_stride)
        self.bn_in = BatchNorm2d(bt_channels)

        self.conv_bt = Conv2d(rng, (bt_channels, bt_channels, 3, 3))
        self.bn_bt = BatchNorm2d(bt_channels)

        self.conv_out = Conv2d(rng, (out_channels, bt_channels, 1, 1))
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

    stage1_bt1: Module
    stage1_bt2: Module
    stage1_bt3: Module

    stage2_bt1: Module
    stage2_bt2: Module
    stage2_bt3: Module
    stage2_bt4: Module

    stage3_bt1: Module
    stage3_bt2: Module
    stage3_bt3: Module
    stage3_bt4: Module
    stage3_bt5: Module
    stage3_bt6: Module

    stage4_bt1: Module
    stage4_bt2: Module
    stage4_bt3: Module

    head: Module

    def __init__(self,
        rng,
        ):

        self.stem = ResStem(rng, (64, 3, 7, 7), stride=2)

        self.stage1_bt1 = BottleneckBlock(rng, 64, 64, 256)
        self.stage1_bt2 = BottleneckBlock(rng, 256, 64, 256)
        self.stage1_bt3 = BottleneckBlock(rng, 256, 64, 256)

        self.stage2_bt1 = BottleneckBlock(rng, 256, 128, 512, in_stride=2)
        self.stage2_bt2 = BottleneckBlock(rng, 512, 128, 512)
        self.stage2_bt3 = BottleneckBlock(rng, 512, 128, 512)
        self.stage2_bt4 = BottleneckBlock(rng, 512, 128, 512)

        self.stage3_bt1 = BottleneckBlock(rng, 512, 256, 1024, in_stride=2)
        self.stage3_bt2 = BottleneckBlock(rng, 1024, 256, 1024)
        self.stage3_bt3 = BottleneckBlock(rng, 1024, 256, 1024)
        self.stage3_bt4 = BottleneckBlock(rng, 1024, 256, 1024)
        self.stage3_bt5 = BottleneckBlock(rng, 1024, 256, 1024)
        self.stage3_bt6 = BottleneckBlock(rng, 1024, 256, 1024)

        self.stage4_bt1 = BottleneckBlock(rng, 1024, 512, 2048, in_stride=2)
        self.stage4_bt2 = BottleneckBlock(rng, 2048, 512, 2048)
        self.stage4_bt3 = BottleneckBlock(rng, 2048, 512, 2048)

        self.head = ResHead(rng, 2048, 1000)
    
    def forward(self, x):
        x = self.stem(x)

        x = self.stage1_bt1(x)
        x = self.stage1_bt2(x)
        x = self.stage1_bt3(x)

        x = self.stage2_bt1(x)
        x = self.stage2_bt2(x)
        x = self.stage2_bt3(x)
        x = self.stage2_bt4(x)

        x = self.stage3_bt1(x)
        x = self.stage3_bt2(x)
        x = self.stage3_bt3(x)
        x = self.stage3_bt4(x)
        x = self.stage3_bt5(x)
        x = self.stage3_bt6(x)

        x = self.stage4_bt1(x)
        x = self.stage4_bt2(x)
        x = self.stage4_bt3(x)

        x = self.head(x)

        return x