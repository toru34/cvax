import jax

from nmax import Module, ModuleTuple
from cvax.models.resnet.blocks import ResStem, ResHead, BottleneckBlock


class ResNet50(Module):
    
    stem: Module

    stage1: Module
    stage2: Module
    stage3: Module
    stage4: Module
    
    head: Module

    def __init__(self, rng, out_dim: int = 1000):

        rngs = jax.random.split(rng, num=18)

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