import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Float


@jit
def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, jnp.eye(P.shape[-1]))


@eqx.filter_jit
def householder(v: jnp.ndarray, D: int) -> Float[Array, "D D"]:
    sign = jnp.sign(v[0])
    k = len(v)
    v = v.at[0].add(sign * jnp.linalg.norm(v))
    u = v / jnp.linalg.norm(v)
    tilde_H = -sign * (jnp.eye(k, dtype=jnp.double) - 2 * jnp.outer(u, u.T))
    if k == D:
        H = tilde_H
    else:
        H = jax.scipy.linalg.block_diag(jnp.eye(D - k), tilde_H)
    return H


@eqx.filter_jit
def orthogonal_matrix(vs: list[jnp.ndarray], D: int, Q: int) -> Float[Array, "D Q"]:
    U = jnp.eye(D, dtype=jnp.double)
    for i, v in enumerate(vs):
        H_n = householder(v, D)
        U = U @ H_n
    return U[:, :Q]
