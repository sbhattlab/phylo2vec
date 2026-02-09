"""Tree balance metrics"""

from phylo2vec import _phylo2vec_core as core


def sackin(vector_or_matrix):
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.sackin(v)


def b2(vector_or_matrix):
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.b2(v)


def leaf_depth_variance(vector_or_matrix):
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.leaf_depth_variance(v)
