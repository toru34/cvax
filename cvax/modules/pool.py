from typing import Literal

import jax
import jax.numpy as jnp

from nmax import Module


class MaxPool2d(Module):
    def __init__(self,
        kernel_width: int,
        stride: int = 2,
        padding: Literal['VALID', 'SAME'] = 'SAME'
        ):
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_dimensions=(1, 1, self.kernel_width, self.kernel_width), window_strides=(1, 1, self.stride, self.stride), padding=self.padding) # TODO: Replace padding