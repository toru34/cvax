import jax.numpy as jnp

from nmax import Parameter, Module
from cvax.utils import config

class BatchNorm2d(Module):

    gamma: Parameter
    beta: Parameter

    def __init__(self, n_channels):
        self.gamma = jnp.ones((1, n_channels, 1, 1))
        self.beta = jnp.zeros((1, n_channels, 1, 1))

        self.momentum = 0.1

        self.mean = jnp.zeros((1, n_channels, 1, 1))
        self.var = jnp.ones((1, n_channels, 1, 1))
    
    def forward(self, x):
        if self._mode == 'eval':
            x_mean = self.mean
            x_var = self.var
        elif self._mode == 'train':
            x_mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
            x_var = jnp.var(x, axis=(0, 2, 3), keepdims=True)

            # Update population statistics
            self.mean = self.momentum * x_mean + (1 - self.momentum) * self.mean
            self.var = self.momentum * x_var + (1 - self.momentum) * self.var

        x_normalised = (x - x_mean) / jnp.sqrt(x_var + config.EPSILON)

        return self.gamma * x_normalised + self.beta