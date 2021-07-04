import jax.nn as nn
import jax.numpy as jnp

from nmax import Module


class Mish(Module):

    def forward(self, x):

        x = x * jnp.tanh(nn.softplus(x))

        return x