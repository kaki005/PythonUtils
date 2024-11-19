import jax.numpy as jnp
from jax import jit


@jit
def log_universal(x):
    """calc universal code length log*(x)"""
    if x == 0:
        return 0
    return 2.0 * jnp.log2(x) + 1
