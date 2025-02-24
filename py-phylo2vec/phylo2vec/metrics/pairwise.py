import numpy as np

from phylo2vec.base.to_newick import _get_ancestry
from phylo2vec.utils.validation import check_v
from phylo2vec import _phylo2vec_core

def cophenetic_distances(v, unrooted=False):
    return _phylo2vec_core.cophenetic_distances(v, unrooted)


PAIRWISE_DISTANCES = {"cophenetic": cophenetic_distances}


def pairwise_distances(v, metric="cophenetic"):
    check_v(v)

    func = PAIRWISE_DISTANCES[metric]

    return func(v)
