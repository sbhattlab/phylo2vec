"""Base class for all optimisation methods in Phylo2Vec."""

import multiprocessing
import random
import time

from dataclasses import dataclass
from typing import Dict, List, Final

import numpy as np

from phylo2vec.datasets import read_fasta
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
    v_opt : numpy.ndarray
        Optimized phylo2vec vector.
    label_mapping : Dict[int, str]
        Mapping of leaf labels (integer) to taxa.
    best_score : float
        The best score achieved during optimization.
    scores : List[float]
        List of scores obtained during optimization.
    """

    v: np.ndarray
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

    def __init__(self, random_seed=None, verbose=False, n_jobs=None):
        self.random_seed = (
            random.randint(0, MAX_SEED) if random_seed is None else random_seed
        )
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.verbose = verbose

        self.n_jobs = self._infer_n_jobs(n_jobs)

    @staticmethod
    def _infer_n_jobs(n_jobs=None):
        return n_jobs or max(MIN_N_JOBS, DEFAULT_N_JOBS)

    @staticmethod
    def _make_label_mapping(records):
        label_mapping = dict(enumerate(r.id.replace(" ", ".") for r in records))

        return label_mapping

    def fit(self, fasta_path) -> BaseResult:
        """Fit an optimizer to a fasta file

        Parameters
        ----------
        fasta_path : str
            Path to fasta file

        Returns
        -------
        v_opt : numpy.ndarray
            Optimized phylo2vec vector
        label_mapping : List[str]
            Mapping of leaf labels (integer) to taxa
        losses : array-like
            List/Array of collected losses
        """
        # TODO: figure out how to change this when the user selects load fasta
        # Probably an boolean "preloaded"
        # If True, change the fasta path to include the module name
        records = list(read_fasta(fasta_path))

        label_mapping = self._make_label_mapping(records)

        n_leaves = len(label_mapping)

        v_init = sample_vector(n_leaves)

        start_time = time.time()

        result = self._optimise(fasta_path, v_init, label_mapping)

        end_time = time.time()

        if self.verbose:
            print(
                f"Optimisation finished in {end_time - start_time:.2f} seconds "
                f"with {len(result.scores)} loss evaluations."
            )

        return result

    def _optimise(self, fasta_path, v, label_mapping):
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
