import jax
import jax.numpy as jnp

from nmax import Module, ModuleTuple
from cvax.utils import key_generator
from cvax.modules import MaxPool2d

from cvax.models.yolo.blocks import YOLOConvBlock


class YOLOTinyBlock(Module):

    def __init__(self, key, base_filters):

        g = key_generator(key)

        self.conv1 = YOLOConvBlock(next(g), kernel_shape=(2 * base_filters, 2 * base_filters, 3, 3))
        self.conv2 = YOLOConvBlock(next(g), kernel_shape=(base_filters, base_filters, 3, 3))
        self.conv3 = YOLOConvBlock(next(g), kernel_shape=(base_filters, base_filters, 3, 3))
        self.conv4 = YOLOConvBlock(next(g), kernel_shape=(2 * base_filters, 2 * base_filters, 1, 1))

    def forward(self, x):

        x = self.conv1(x)

        x_left = x

        x = jnp.split(x, indices_or_sections=2, axis=1)[0]
        x = self.conv2(x)
        x = jnp.concatenate([x, self.conv3(x)], axis=1)
        x = self.conv4(x)

        x = jnp.concatenate([x_left, x], axis=1)

        return x


class YOLOv4Tiny(Module):

    stage1: Module

    def __init__(self, key):

        g = key_generator(key)

        self.stem = ModuleTuple((
            YOLOConvBlock(next(g), kernel_shape=(32, 3, 3, 3), stride=2),
            YOLOConvBlock(next(g), kernel_shape=(64, 32, 3, 3), stride=2),
        ))

        self.stage1 = YOLOTinyBlock(next(g), base_filters=32)
        self.pool1 = MaxPool2d(kernel_width=2, stride=2)

        self.stage2 = YOLOTinyBlock(next(g), base_filters=64)
        self.pool2 = MaxPool2d(kernel_width=2, stride=2)

        self.stage3 = YOLOTinyBlock(next(g), base_filters=128)
        self.pool3 = MaxPool2d(kernel_width=2, stride=2)

        self.stage4a = ModuleTuple((
            YOLOConvBlock(next(g), kernel_shape=(512, 512, 3, 3)),
            YOLOConvBlock(next(g), kernel_shape=(256, 512, 1, 1)),
        ))

        self.stage5a = ModuleTuple((
            YOLOConvBlock(next(g), kernel_shape=(512, 256, 3, 3)),
            YOLOConvBlock(next(g), kernel_shape=(255, 512, 1, 1), batch_norm=False, activation=False),
        ))

        self.stage4b = YOLOConvBlock(next(g), kernel_shape=(128, 256, 1, 1))
        self.stage5b = ModuleTuple((
            YOLOConvBlock(next(g), kernel_shape=(256, 384, 3, 3)),
            YOLOConvBlock(next(g), kernel_shape=(255, 256, 1, 1), batch_norm=False, activation=False),
        ))
    
    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        
        xa = x
        xb1 = jnp.split(x, indices_or_sections=2, axis=1)[1]

        xa = self.pool3(xa)

        xa = self.stage4a(xa)
        xb2 = xa
        xa = self.stage5a(xa)

        xb2 = self.stage4b(xb2)
        xb2 = jax.image.resize(xb2, jnp.array(xb2.shape) * 2, method='nearest')
        xb2 = xb2[::2, ::2] # TODO

        xb = jnp.concatenate([xb1, xb2], axis=1)
        xb = self.stage5b(xb)

        return xa, xb