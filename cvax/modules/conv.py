import math
from typing import Literal

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

from nmax import Module, Parameter


class Conv2d(Module):

    W: Parameter
    b: Parameter

    def __init__(self,
        key,
        kernel_shape: tuple[int, int, int, int],
        stride: int = 1,
        padding: Literal["VALID", "SAME"] = "SAME",
        use_bias: bool = True,
        activation = lambda x: x,
        ):

        self.W = init.he_normal()(key, kernel_shape)

        if use_bias:
            self.b = jnp.zeros((1, kernel_shape[0], 1, 1))
        
        self.stride = stride
        self.padding = padding
        self.activation = activation
    
    def forward(self, x):
        x = jax.lax.conv_general_dilated(
            x,
            self.W,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
        ) # TODO: Fix padding

        if hasattr(self, 'b'):
            x += self.b
        
        return self.activation(x)


class DepthwiseConv2d(Module):

    W: Parameter
    b: Parameter

    def __init__(self,
        key,
        kernel_shape: tuple[int, int, int, int],
        stride: int = 1,
        padding: Literal["VALID", "SAME"] = "SAME",
        use_bias: bool = True,
        activation = lambda x: x,
        ):

        self.W = init.he_normal()(key, kernel_shape)

        if use_bias:
            self.b = jnp.zeros(shape=(1, kernel_shape[0], 1, 1))
        
        self.stride = stride
        self.padding = padding
        self.activation = activation

    def forward(self, x):
        x = jax.lax.conv_general_dilated(
            x,
            self.W,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            feature_group_count=self.W.shape[0]
        ) # TODO: Fix padding

        if hasattr(self, 'b'):
            x += self.b
        
        return self.activation(x)