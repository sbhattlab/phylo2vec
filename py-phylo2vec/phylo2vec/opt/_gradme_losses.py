"""Loss functions for GradME optimisation."""

import jax.numpy as jnp

from jax import jit, lax
from jax.nn import softmax
from jax.scipy.special import logsumexp


@jit
def get_edges_exp_log(W, rooted):
    """Log version of get_edges_exp

    Parameters
    ----------
    W : jax.numpy.array
        Tree distribution
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    E : jax.numpy.array
        Log of the expected objective value of a tree drawn with distribution W
    """
    # Add jnp.finfo(float).eps to W.tmp to avoid floating point errors with float32
    W_tmp = (
        jnp.pad(W, (0, 1), constant_values=jnp.finfo(float).eps) + jnp.finfo(float).eps
    )

    n_leaves = len(W) + 1

    E = jnp.zeros((n_leaves, n_leaves))

    trindx_x, trindx_y = jnp.tril_indices(n_leaves - 1, -1)

    E = E.at[1, 0].set(
        0.5 * E[0, 0] * W_tmp[0, 0] + jnp.log(0.25 * (2 - rooted) * W_tmp[0, 0])
    )

    E = E + E.T

    def body(carry, _):
        E, i = carry

        E_new = jnp.zeros((n_leaves, n_leaves))

        ##### j for loop #####
        trindx_x_i = jnp.where(trindx_x < i, trindx_x, 1)
        trindx_y_i = jnp.where(trindx_x < i, trindx_y, 0)

        index = (trindx_x_i, trindx_y_i)

        E_new = E_new.at[index].set(
            E[index]
            + jnp.log(
                1
                - 0.5 * (W_tmp[i - 1, index[1]] + W_tmp[i - 1, index[0]])
                + jnp.finfo(float).eps
            )
        )
        ##### End j for loop #####

        ##### k for loop #####
        # exp array
        mask_Ei = jnp.where(jnp.arange(n_leaves) >= i, 0, 1)
        exp_array = E * jnp.where(jnp.arange(n_leaves) >= i, 0, mask_Ei.T)

        # coef array
        mask_Wi = jnp.where(jnp.arange(n_leaves) >= i, 0, 0.5 * W_tmp[i - 1])
        coef_array = (jnp.zeros_like(W_tmp) + mask_Wi).at[:, i].set(
            0.25 * W_tmp[i - 1]
        ) * (1 - jnp.eye(W_tmp.shape[0]))

        # logsumexp
        tmp = logsumexp(exp_array, b=coef_array, axis=-1) * mask_Ei

        E_new = E_new.at[i, :].set(tmp)

        # Update E
        E = E_new + E_new.T

        ##### End k for loop #####

        return (E, i + 1), None

    # https://github.com/google/jax/issues/5454
    (E, _), _ = lax.scan(body, (E, 2), None, length=n_leaves - 2)

    return E


@jit
def gradme_loss(W, D, rooted):
    """Log version of the BME loss function used in GradME

    Parameters
    ----------
    W : jax.numpy.array
        Ordered tree probability matrix
    D : jax.numpy.array
        Distance matrix
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    float
        Continuous BME loss
    """
    W = softmax(W, where=jnp.tril(jnp.ones_like(W)))

    E = get_edges_exp_log(W, rooted)

    loss = logsumexp(E, b=D)

    return loss
