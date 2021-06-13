from nmax import Module, ModuleTuple
from cvax.utils import key_generator
from cvax.modules import MaxPool2d
from cvax.models.yolo.blocks import YOLOConvBlock


class YOLOv2Tiny(Module):

    def __init__(self, key):

        g = key_generator(key)

        self.modules = ModuleTuple((
            YOLOConvBlock(next(g), kernel_shape=(16, 3, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
            YOLOConvBlock(next(g), kernel_shape=(32, 16, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
            YOLOConvBlock(next(g), kernel_shape=(64, 32, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
            YOLOConvBlock(next(g), kernel_shape=(128, 64, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
            YOLOConvBlock(next(g), kernel_shape=(256, 128, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
            YOLOConvBlock(next(g), kernel_shape=(512, 256, 3, 3)),
            MaxPool2d(kernel_width=2, stride=1),
            YOLOConvBlock(next(g), kernel_shape=(1024, 512, 3, 3)),
            YOLOConvBlock(next(g), kernel_shape=(512, 1024, 3, 3)),
            YOLOConvBlock(next(g), kernel_shape=(425, 512, 1, 1), batch_norm=False, activation=False),
        ))
    
    def forward(self, x):

        x = self.modules(x)

        return x
