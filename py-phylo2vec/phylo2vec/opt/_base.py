"""Base class for all optimisation methods in Phylo2Vec."""

import multiprocessing
import random
import time
from dataclasses import dataclass
from typing import Dict, Final, List

import numpy as np

from phylo2vec.datasets import read_fasta
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector

# Multiprocessing
DEFAULT_N_JOBS: Final = multiprocessing.cpu_count() // 4
MIN_N_JOBS: Final = 4
# Seeding
MAX_SEED = 42


@dataclass
class BaseResult:
    """Result of the optimization process.

    Attributes
    ----------
    best : numpy.ndarray
        Optimized phylo2vec vector or matrix.
    label_mapping : Dict[int, str]
        Mapping of leaf labels (integer) to taxa.
    best_score : float
        The best score achieved during optimization.
    scores : List[float]
        List of scores obtained during optimization.
    """

    best: np.ndarray
    label_mapping: Dict[int, str]
    best_score: float
    scores: List[float]


class BaseOptimizer:
    """
    Base class for all phylo2vec-based optimizers

    Parameters
    ----------
    random_seed : int, optional
        Random seed, by default None
    """

    def __init__(self, mode="vector", random_seed=None, verbose=False, n_jobs=None):
        if mode not in ("vector", "matrix"):
            raise ValueError("Mode must be either 'vector' or 'matrix'.")

        self.mode = mode

        self.random_seed = random_seed or random.randint(0, MAX_SEED)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.verbose = verbose

        self.n_jobs = self._infer_n_jobs(n_jobs)

    @staticmethod
    def _infer_n_jobs(n_jobs=None):
        return n_jobs or max(MIN_N_JOBS, DEFAULT_N_JOBS)

    def fit(self, fasta_path) -> BaseResult:
        """Fit an optimizer to a fasta file

        Parameters
        ----------
        fasta_path : str
            Path to fasta file

        Returns
        -------
        result : BaseResult
            Result of the optimization process, containing the optimized vector,
            label mapping, best score, and scores during optimization.
        """
        # TODO: figure out how to change this when the user selects load fasta
        # Probably an boolean "preloaded"
        # If True, change the fasta path to include the module name
        records = read_fasta(fasta_path)

        # Make a label mapping from the records
        label_mapping = dict(enumerate(r.id.replace(" ", ".") for r in records))

        n_leaves = len(label_mapping)

        if self.mode == "vector":
            obj_init = sample_vector(n_leaves)
        elif self.mode == "matrix":
            obj_init = sample_matrix(n_leaves)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        start_time = time.time()

        result = self._optimise(fasta_path, obj_init, label_mapping)

        end_time = time.time()

        if self.verbose:
            print(
                f"Optimisation finished in {end_time - start_time:.2f} seconds "
                f"with {len(result.scores)} loss evaluations."
            )

        return result

    def _optimise(self, fasta_path, tree, label_mapping):
        raise NotImplementedError

    def __repr__(self):
        # TODO: maybe something like sklearn pprint
        # https://github.com/scikit-learn/scikit-learn/blob/093e0cf14aff026cca6097e8c42f83b735d26358/sklearn/utils/_pprint.py#L116
        format_string = f"{self.__class__.__name__}("

        for item in vars(self):
            format_string += "\n"
            # TODO: pprint if dict?
            format_string += f"\t{item}={repr(self.__getattribute__(item))},"

        format_string = format_string[:-1] + "\n)"

        return format_string
