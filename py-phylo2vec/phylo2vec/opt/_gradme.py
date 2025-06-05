"""Methods for GradME optimisation."""

from dataclasses import dataclass

import jax.numpy as jnp
import optax
import rpy2
import rpy2.robjects as ro

from jax import jit, value_and_grad
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from tqdm import tqdm

from phylo2vec.opt._base import BaseOptimizer, BaseResult
from phylo2vec.opt._gradme_losses import gradme_loss
from phylo2vec.utils.vector import queue_shuffle, reroot_at_random


# Disable rpy2 warning
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None


@dataclass
class GradMEResult(BaseResult):
    """Result of the GradME optimization.

    See BaseResult for more details.

    Attributes
    ----------
    W : jax.numpy.ndarray
        The optimized weight matrix representing the phylogenetic tree.
    """

    W: jnp.ndarray


class GradMEOptimizer(BaseOptimizer):
    """GradME Optimizer for phylogenetic trees.

    This optimizer uses the GradME algorithm to optimize phylogenetic trees.
    It computes the loss using the GradME loss function and updates the tree
    representation accordingly.

    Parameters
    ----------
    random_seed : int, optional
        Random seed for reproducibility, by default None
    n_jobs : int, optional
        Number of parallel jobs, by default None
    verbose : bool, optional
        Verbosity level, by default False
    """

    def __init__(
        self,
        model,
        solver="adafactor",
        learning_rate=1.5,
        rooted=False,
        n_shuffles=100,
        n_iter_per_step=5000,
        tol=1e-8,
        random_seed=None,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(random_seed=random_seed, n_jobs=n_jobs, verbose=verbose)

        self.model = model

        self.optimizer = getattr(optax, solver)(learning_rate=learning_rate)
        self.learning_rate = learning_rate
        self.rooted = rooted
        self.n_shuffles = n_shuffles
        self.n_iter_per_step = n_iter_per_step
        self.tol = tol

    def _optimise(
        self,
        fasta_path,
        v,
        label_mapping,
    ):
        data = self.pdist(fasta_path, self.model)
        dm = jnp.asarray(data)
        k = dm.shape[0] - 1

        # Forward and backward pass function
        value_and_grad_fun = jit(value_and_grad(gradme_loss))

        # Initial "best" score, set as an arbitrarily high value
        best_score = 1e8

        # List of scores obtained during optimization
        scores = []

        iterator = range(self.n_shuffles)

        if self.verbose:
            iterator = tqdm(iterator)

        for _ in iterator:
            w_in = self._init_W(k)

            w_out = self._step(
                w_in,
                dm,
                value_and_grad_fun,
            )

            v = w_out.argmax(1)

            w_discrete = jnp.eye(w_out.shape[0])[v]

            score = gradme_loss(w_discrete, dm, rooted=True)

            best_score = min(best_score, score)

            scores.append(best_score)

            if not self.rooted:
                v = reroot_at_random(v)

            # Queue shuffle
            _, vec_mapping = queue_shuffle(v, shuffle_cherries=True)

            # Re-arrange the label mapping and the distance matrix
            col_order = []
            for i, idx in enumerate(vec_mapping):
                label_mapping[i] = label_mapping[idx]
                col_order.append(label_mapping[i])

            dm = jnp.asarray(data.loc[col_order, col_order])

            if self.verbose:
                iterator.set_postfix({"\033[95m Best score ": best_score})

        v = jnp.eye(w_out.shape[0])[w_out.argmax(1)]

        best_params = GradMEResult(
            v=v,
            best_score=best_score,
            scores=scores,
            W=w_out,
            label_mapping=label_mapping,
        )

        return best_params

    @staticmethod
    def _init_W(k, eps=1e-8):
        x = jnp.tril(jnp.ones((k, k)))

        w_init = x / (x.sum(1)[:, jnp.newaxis] + eps)

        return w_init

    def _step(self, w, dm, value_and_grad_fun):
        state = self.optimizer.init(w)

        prev_loss = 1e8

        for _ in range(self.n_iter_per_step):
            loss, gradients = value_and_grad_fun(w, dm, self.rooted)

            if jnp.abs(loss - prev_loss) < self.tol:
                break

            prev_loss = loss

            updates, state = self.optimizer.update(gradients, state, w)

            w = optax.apply_updates(w, updates)

        return w

    @staticmethod
    def pdist(fasta_path, model):
        with localconverter(ro.default_converter + pandas2ri.converter):
            importr("ape")

            ro.globalenv["fasta_path"] = fasta_path
            ro.globalenv["model"] = model

            # DNA Evolution model: F81 + Gamma
            dm = ro.r(
                """
                aln <- read.FASTA(fasta_path, type = "DNA")

                dm <- dist.dna(aln, model = model)

                D <- as.data.frame(as.matrix(dm))
                D
                """
            )

        return dm
