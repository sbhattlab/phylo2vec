"""Phylo2Vec: A Python package for phylogenetic vector representations."""

from phylo2vec._phylo2vec_core import __version__  # pylint: disable=no-name-in-module

from phylo2vec.base.ancestry import from_ancestry, to_ancestry
from phylo2vec.base.edges import from_edges, to_edges
from phylo2vec.base.newick import from_newick, to_newick
from phylo2vec.base.pairs import from_pairs, to_pairs
from phylo2vec.io.reader import load, load_newick
from phylo2vec.io.writer import save, save_newick
from phylo2vec import stats
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector

__all__ = [
    "__version__",
    "from_ancestry",
    "from_edges",
    "from_newick",
    "from_pairs",
    "load",
    "load_newick",
    "sample_matrix",
    "sample_vector",
    "save",
    "save_newick",
    "stats",
    "to_ancestry",
    "to_edges",
    "to_newick",
    "to_pairs",
]
