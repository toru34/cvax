import jax.nn as nn

from nmax import Module, ModuleTuple

from cvax.utils import key_generator
from cvax.modules.pool import AveragePool2d
from cvax.modules.conv import Conv2d, DepthwiseConv2d
from cvax.models.mobilenet.blocks import MobileBottleneckBlock


class MobileNetV2(Module):

    stem: Module
    blocks: Module

    def __init__(self,
        key,
        ):

        g = key_generator(key)

        self.stem = ModuleTuple((
            Conv2d(next(g), kernel_shape=(48, 3, 3, 3), stride=2, activation=nn.relu6),
            DepthwiseConv2d(next(g), kernel_shape=(48, 1, 3, 3), activation=nn.relu6),
            Conv2d(next(g), kernel_shape=(24, 48, 1, 1)),
        ))

        self.blocks = ModuleTuple((
            MobileBottleneckBlock(next(g), in_channels=24, out_channels=32, stride=2),
            MobileBottleneckBlock(next(g), in_channels=32, out_channels=32, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=32, out_channels=48, stride=2),
            MobileBottleneckBlock(next(g), in_channels=48, out_channels=48, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=48, out_channels=48, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=48, out_channels=88, stride=2),
            MobileBottleneckBlock(next(g), in_channels=88, out_channels=88, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=88, out_channels=88, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=88, out_channels=88, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=88, out_channels=136),
            MobileBottleneckBlock(next(g), in_channels=136, out_channels=136, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=136, out_channels=136, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=136, out_channels=224, stride=2),
            MobileBottleneckBlock(next(g), in_channels=224, out_channels=224, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=224, out_channels=224, use_residual=True),
            MobileBottleneckBlock(next(g), in_channels=224, out_channels=448),
        ))

        self.head = ModuleTuple((
            Conv2d(next(g), kernel_shape=(1792, 448, 1, 1), activation=nn.relu6),
            AveragePool2d(),
            Conv2d(next(g), kernel_shape=(1001, 1792, 1, 1)),
        ))
    
    def forward(self, x):

        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        x = x.reshape(-1, 1001)
        x = nn.softmax(x, axis=-1)

        return x