import jax

def key_generator(rng):
    while True:
        rng, sub_rng = jax.random.split(rng)
        yield sub_rng