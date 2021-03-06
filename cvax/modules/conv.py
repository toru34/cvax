import math
from typing import Literal

import jax
import jax.numpy as jnp

from nmax import Module, Parameter

class Conv2d(Module):

    W: Parameter
    b: Parameter

    def __init__(self,
        rng,
        kernel_shape: tuple[int, int, int, int],
        stride: int = 1,
        padding: Literal["VALID", "SAME"] = "SAME",
        use_bias: bool = True,
        ):
        self.W = jax.random.normal(rng, kernel_shape) # TODO
        
        self.use_bias = use_bias
        if self.use_bias:
            self.b = jnp.zeros((1, kernel_shape[0], 1, 1))
        
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        x = jax.lax.conv(x, self.W, window_strides=(self.stride, self.stride), padding=self.padding) # TODO: Fix padding

        if self.use_bias:
            return x + self.b
        else:
            return x