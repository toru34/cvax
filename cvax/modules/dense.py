import jax
import jax.numpy as jnp
import jax.nn.initializers as init

from nmax import Parameter, Module

class Dense(Module):
    
    W: Parameter
    b: Parameter

    def __init__(self, rng, in_dim, out_dim):
        # self.W = jax.random.normal(rng, (in_dim, out_dim))
        self.W = init.he_normal()(rng, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim)
    
    def forward(self, x):
        return x @ self.W + self.b