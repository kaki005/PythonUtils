import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import cholesky, inv


@jit
def PSD_inv(matrix: jnp.ndarray) -> jnp.ndarray:
    """inverse of Positive semi-Definite matrix. (O(N^2))

    Args:
        matrix (jnp.ndarray): PSD matrix

    Returns:
        jnp.ndarray: inverse of PSD matrix
    """
    L = cholesky(matrix)
    Linv = inv(L)
    return Linv.T @ Linv
