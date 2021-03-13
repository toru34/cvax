import jax

from nmax import Module, ModuleTuple
from cvax.models.resnet.blocks import ResStem, ResHead, BottleneckBlock


class ResNet101(Module):

    stem: Module

    stage1: Module
    stage2: Module
    stage3: Module
    stage4: Module

    head: Module

    def __init__(self, rng, out_dim: int = 1000):

        g = key_generator(rng)

        self.stem = ResStem(next(g), (64, 3, 7, 7), stride=2)

        self.stage1 = ModuleTuple((
            BottleneckBlock(next(g), 64, 64, 256),
            BottleneckBlock(next(g), 256, 64, 256),
            BottleneckBlock(next(g), 256, 64, 256),
        ))

        self.stage2 = ModuleTuple((
            BottleneckBlock(next(g), 256, 128, 512, in_stride=2),
            BottleneckBlock(next(g), 512, 128, 512),
            BottleneckBlock(next(g), 512, 128, 512),
            BottleneckBlock(next(g), 512, 128, 512),
        ))

        self.stage3 = ModuleTuple((
            BottleneckBlock(next(g), 512, 256, 1024, in_stride=2),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
            BottleneckBlock(next(g), 1024, 256, 1024),
        ))

        self.stage4 = ModuleTuple((
            BottleneckBlock(next(g), 1024, 512, 2048, in_stride=2),
            BottleneckBlock(next(g), 2048, 512, 2048),
            BottleneckBlock(next(g), 2048, 512, 2048),
        ))

        self.head = ResHead(next(g), 2048, out_dim)
    
    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.head(x)

        return x