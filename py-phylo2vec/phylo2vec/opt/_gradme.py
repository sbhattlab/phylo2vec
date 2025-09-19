"""Methods for GradME optimisation."""

import random

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from jax import jit, value_and_grad
from jax.nn import softmax
from tqdm import tqdm

from phylo2vec.opt._api import register_method
from phylo2vec.opt._base import BaseOptimizer, BaseResult
from phylo2vec.opt._gradme_losses import gradme_loss, make_W
from phylo2vec.opt.utils import dnadist
from phylo2vec.utils.vector import queue_shuffle, reroot


@dataclass
class GradMEResult(BaseResult):
    """Result of the GradME optimization.

    See BaseResult for more details.

    Attributes
    ----------
    best_W : jax.numpy.ndarray
        The optimized weight matrix representing the phylogenetic tree.
    """

    best_W: jnp.ndarray


@register_method()
class GradME(BaseOptimizer):
    """GradME Optimizer for phylogenetic trees.

    This optimizer uses the GradME algorithm to optimize phylogenetic trees.
    It computes the loss using the GradME loss function and updates the tree
    representation accordingly.

    Parameters
    ----------
    model : str
        Phylogenetic model to use (e.g., "JC", "K80", etc.)
    gamma : bool, optional
        Whether to use gamma correction for the distance matrix, by default False
    solver : str, optional
        The optimization solver to use, by default "adafactor"
    learning_rate : float, optional
        Learning rate for the optimizer, by default 1.5
    rooted : bool, optional
        Whether to operate in a rooted tree space or not, by default False
    n_shuffles : int, optional
        Number of shuffles to perform during optimization, by default 100
    n_iter_per_step : int, optional
        Number of iterations per optimization step, by default 5000
    patience : int, optional
        Number of iterations to wait for improvement before stopping, by default 10
    tol : float, optional
        Tolerance for convergence, by default 1e-8
    random_seed : int, optional
        Random seed for reproducibility, by default None
    n_jobs : int, optional
        Number of parallel jobs, by default None
    verbose : bool, optional
        Verbosity level, by default False
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer, by default {}
    """

    def __init__(
        self,
        model,
        gamma=False,
        solver="adafactor",
        learning_rate=1.5,
        rooted=False,
        n_shuffles=100,
        n_iter_per_step=5000,
        patience=10,
        tol=1e-8,
        random_seed=None,
        n_jobs=None,
        verbose=False,
        **optimizer_kwargs,
    ):
        super().__init__(
            mode="vector", random_seed=random_seed, n_jobs=n_jobs, verbose=verbose
        )

        self.model = model
        self.gamma = gamma
        self.solver = solver
        self.learning_rate = learning_rate
        self.rooted = rooted
        self.n_shuffles = n_shuffles
        self.n_iter_per_step = n_iter_per_step
        self.patience = patience
        self.tol = tol
        self.optimizer_kwargs = optimizer_kwargs

    def _optimise(
        self,
        fasta_path,
        tree,
        label_mapping,
    ) -> GradMEResult:
        """Optimise a tree using GradME

        Parameters
        ----------
        fasta_path : str
            Path to fasta file
        tree : numpy.ndarray
            random tree to optimize, in v representation (unused)
        label_mapping : Dict[int, str]
            Current mapping of leaf labels (integer) to taxa

        Returns
        -------
        GradMEResult
            best : numpy.ndarray
                Optimized phylo2vec vector.
            best_W : jax.numpy.ndarray
                The optimized weight matrix representing the phylogenetic tree.
            label_mapping : Dict[int, str]
                Mapping of leaf labels (integer) to taxa.
            best_score : float
                The best score achieved during optimization.
            scores : List[float]
                List of scores obtained during optimization.
        """
        data = dnadist(fasta_path, self.model, self.gamma)
        dm = jnp.asarray(data)
        n_leaves = dm.shape[0]

        # Forward and backward pass function
        optimizer = getattr(optax, self.solver)(
            learning_rate=self.learning_rate, **self.optimizer_kwargs
        )
        value_and_grad_fun = jit(value_and_grad(gradme_loss))

        # Initial "best" score, set as an arbitrarily high value
        best_score = 1e8
        best_W = jnp.tril(jnp.ones((n_leaves, n_leaves)))
        best_tree = jnp.zeros((n_leaves - 1,))
        best_label_mapping = label_mapping.copy()

        # List of scores obtained during optimization
        scores = []

        iterator = range(self.n_shuffles)

        if self.verbose:
            iterator = tqdm(iterator)

        # Patience
        wait = 0

        for _ in iterator:
            # Initialize parameters for the current iteration
            params_in = self._init(n_leaves)

            # Perform a round of optimization
            params_out = self._step(
                dm=dm,
                params=params_in,
                value_and_grad_fun=value_and_grad_fun,
                optimizer=optimizer,
            )

            # Transform the parameters into a weight matrix
            # Compute the loss of the most probable tree
            w_out = make_W(params_out, n_leaves)
            tree = w_out.argmax(1)
            score = gradme_loss(jnp.eye(n_leaves - 1)[tree], dm, rooted=self.rooted)

            # Update the best score and the best parameters
            if score < best_score:
                best_tree = tree
                best_W = w_out
                best_score = score
                best_label_mapping = label_mapping.copy()
                wait = 0
            else:
                wait += 1

            scores.append(best_score)

            dm, label_mapping = self._shuffle(tree, data, label_mapping, self.rooted)

            if self.verbose:
                iterator.set_postfix_str(f"Current loss: {best_score:.6f}")

            if wait >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping after {self.patience} iterations "
                        f"without improvement."
                    )
                break

        return GradMEResult(
            best=best_tree,
            best_W=softmax(best_W, where=jnp.tril(jnp.ones_like(best_W))),
            best_score=best_score,
            scores=scores,
            label_mapping=best_label_mapping,
        )

    def _step(self, dm, params, value_and_grad_fun, optimizer) -> jax.numpy.ndarray:
        """Perform a step of the hill-climbing optimisation

        Parameters
        ----------
        dm : jax.numpy.ndarray
            Distance matrix of the taxa
        params : jax.numpy.ndarray
            Flattened triangular matrix representing the continuous tree
        value_and_grad_fun : Callable
            Function to compute the loss and gradients
        optimizer : optax optimizer
            Optimizer to use for gradient descent

        Returns
        -------
        params : jax.numpy.ndarray
            Updated parameters after one step of optimization.
        """
        params = self._init(n_leaves=dm.shape[0])

        state = optimizer.init(params)

        prev_loss = 1e8

        for _ in range(self.n_iter_per_step):
            loss, gradients = value_and_grad_fun(params, dm, self.rooted)

            if jnp.abs(loss - prev_loss) < self.tol:
                break

            prev_loss = loss

            updates, state = optimizer.update(gradients, state, params)
            params = optax.apply_updates(params, updates)

        return params

    @staticmethod
    def _init(n_leaves: int) -> jax.numpy.ndarray:
        """Initialise a set of parameters for n_leaves

        Parameters
        ----------
        n_leaves : int
            Number of leaves

        Returns
        -------
        jax.numpy.ndarray
            Initial parameters as a flattened triangular matrix
            representing the continuous tree.
        """
        key = jax.random.PRNGKey(0)

        length = n_leaves * (n_leaves - 1) // 2

        return (
            0.5 * jnp.ones(length)
            + jax.random.normal(key, shape=(length,)) * 1 / n_leaves
        )

    @staticmethod
    def _shuffle(
        tree, data, label_mapping, rooted
    ) -> Tuple[jax.numpy.ndarray, Dict[int, str]]:
        """
        Shuffle the tree
        (and the label mapping and data) accordingly

        Parameters
        ----------
        tree : jax.numpy.ndarray
            The tree represented as a vector
        data : pandas.DataFrame
            Distance matrix of the taxa
        label_mapping : Dict[int, str]
            Current mapping of leaf labels (integer) to taxa
        rooted : bool
            Whether the tree is rooted or not

        Returns
        -------
        dm : jax.numpy.ndarray
            The shuffled distance matrix.
        label_mapping : Dict[int, str]
            The updated mapping of leaf labels (integer) to taxa after shuffling.
        """
        if not rooted:
            root = random.randint(0, data.shape[0])
            tree = reroot(tree, root)

        # Queue shuffle
        _, vec_mapping = queue_shuffle(tree, shuffle_cherries=False)

        # Re-arrange the label mapping and the distance matrix
        label_mapping_new = {i: label_mapping[idx] for i, idx in enumerate(vec_mapping)}
        label_mapping = label_mapping_new

        col_order = [label_mapping_new[i] for i in range(len(label_mapping_new))]

        dm = jnp.asarray(data.loc[col_order, col_order])

        return dm, label_mapping


# This is a compatibility alias for the HillClimbing class.
GradMEOptimizer = GradME
