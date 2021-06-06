from nmax import Module, ModuleTuple
from cvax.utils import key_generator
from cvax.modules import MaxPool2d
from cvax.models.yolo.blocks import YOLOConvBlock

class YOLOv1(Module):

    stage1: Module
    stage2: Module
    stage3: Module
    stage4: Module
    stage5: Module

    def __init__(self, key):

        g = key_generator(key)

        self.stage1 = ModuleTuple((
            YOLOConvBlock(next(g), (64, 3, 7, 7), stride=2),
            MaxPool2d(kernel_width=2, stride=2),
        ))

        self.stage2 = ModuleTuple((
            YOLOConvBlock(next(g), (192, 64, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
        ))

        self.stage3 = ModuleTuple((
            YOLOConvBlock(next(g), (128, 192, 1, 1)),
            YOLOConvBlock(next(g), (256, 128, 3, 3)),
            YOLOConvBlock(next(g), (256, 256, 1, 1)),
            YOLOConvBlock(next(g), (512, 256, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
        ))

        self.stage4 = ModuleTuple((
            YOLOConvBlock(next(g), (256, 512, 1, 1)),
            YOLOConvBlock(next(g), (512, 256, 3, 3)),
            YOLOConvBlock(next(g), (256, 512, 1, 1)),
            YOLOConvBlock(next(g), (512, 256, 3, 3)),
            YOLOConvBlock(next(g), (256, 512, 1, 1)),
            YOLOConvBlock(next(g), (512, 256, 3, 3)),
            YOLOConvBlock(next(g), (256, 512, 1, 1)),
            YOLOConvBlock(next(g), (512, 256, 3, 3)),
            YOLOConvBlock(next(g), (512, 512, 1, 1)),
            YOLOConvBlock(next(g), (1024, 512, 3, 3)),
            MaxPool2d(kernel_width=2, stride=2),
        ))

        self.stage5 = ModuleTuple((
            YOLOConvBlock(next(g), (512, 1024, 1, 1)),
            YOLOConvBlock(next(g), (1024, 512, 3, 3)),
            YOLOConvBlock(next(g), (512, 1024, 1, 1)),
            YOLOConvBlock(next(g), (1024, 512, 3, 3)),
            YOLOConvBlock(next(g), (1024, 1024, 3, 3)),
            YOLOConvBlock(next(g), (1024, 1024, 3, 3), stride=2),
            YOLOConvBlock(next(g), (1024, 1024, 3, 3)),
            YOLOConvBlock(next(g), (1024, 1024, 3, 3)),
        ))

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        # TODO: Add dropout

        # TODO: Add local conv2d

        # TODO: Add yolo

        return x