"""Loss functions for GradME optimisation."""

import math

import jax.numpy as jnp

from jax import jit, lax
from jax.nn import softplus
from jax.scipy.special import logsumexp


@jit
def get_edges_exp_log(W, rooted):
    """
    Calculate the expectation of the objective value of a tree drawn with distribution W.

    E_ij = Expected value of the path length between nodes i and j in the tree

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


def make_W(params, n_leaves=None, eps=1e-8):
    """'Un-flatten' params to a W matrix representing the distribution of ordered trees

    Parameters
    ----------
    params : jax.numpy.ndarray
        Flattened version of W
    n_leaves : int, optional
        Number of leaves in the tree, by default None.
        If None, it will be inferred from the length of params.
    eps : float, optional
        Term added to improve numerical stability, by default 1e-8

    Returns
    -------
    W : jax.numpy.ndarray
        distribution of ordered trees
    """
    if n_leaves is None:
        # Solution of quadratic equation: k^2 - k - 2*len(params)
        k = int((1 + math.sqrt(1 + 8 * len(params))) // 2) - 1
    else:
        k = n_leaves - 1

    W = jnp.zeros((k, k)).at[jnp.tril_indices(k)].set(softplus(params))

    W = W / (jnp.tril(W).sum(1)[:, jnp.newaxis] + eps)

    return W


@jit
def _gradme_loss(W, dm, rooted):
    expected_path_lengths = get_edges_exp_log(W, rooted)

    loss = logsumexp(expected_path_lengths, b=dm)

    return loss


def gradme_loss(weights, dm, rooted):
    """Log version of the BME loss function used in GradME

    Parameters
    ----------
    weights : jax.numpy.array
        Can be a squared matrix (ordered tree probability matrix)
        or a vector (flattened tree probability matrix)
    dm : jax.numpy.array
        Distance matrix
    rooted : bool
        True is the tree is rooted, otherwise False

    Returns
    -------
    float
        Continuous BME loss
    """
    if weights.ndim == 1:
        weights = make_W(weights, dm.shape[0])

    if not (weights.ndim == 2 and weights.shape[0] == weights.shape[1]):
        raise ValueError("Input must be a square matrix.")

    return _gradme_loss(weights, dm, rooted)
