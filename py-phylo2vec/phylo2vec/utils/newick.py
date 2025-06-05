"""Newick string manipulation functions"""

import phylo2vec._phylo2vec_core as core


def find_num_leaves(newick: str) -> int:
    """Calculate the number of leaves in a tree from its Newick

    Parameters
    ----------
    newick : str
        Newick representation of a tree

    Returns
    -------
    int
        Number of leaves
    """
    return core.find_num_leaves(newick)


def remove_branch_lengths(newick: str) -> str:
    """Remove branch lengths annotations from a Newick string

    Example: "(((2:0.02,1:0.01),0:0.041),3:1.42);" --> "(((2,1),0),3);"

    Parameters
    ----------
    newick : str
        Newick string

    returns
    ----------
    newick : str
        Newick string without branch lengths
    """
    return core.remove_branch_lengths(newick)


def remove_parent_labels(newick: str) -> str:
    """Remove parent labels from the Newick string

    Example: "(((2,1)4,0)5,3)6;;" --> "(((2,1),0),3);"

    Parameters
    ----------
    newick : str
        Newick representation of a tree

    returns
    ----------
    newick : str
        Newick string without parental/internal labels
    """
    return core.remove_parent_labels(newick)


def create_label_mapping(newick):
    """
    Create an integer-taxon label mapping (label_mapping)
    from a string-based newick (where leaves are strings)
    and produce a mapped integer-based newick (where leaves are integers)
    this also remove annotations pertaining to parent nodes

    Parameters
    ----------
    newick : str
        Newick with string labels

    Returns
    -------
    newick_int : str
        Newick with integer labels
    label_mapping : Dict[int, str]
        Mapping of leaf labels (as integers) to taxa
    """
    newick_int, label_mapping = core.create_label_mapping(newick)

    return newick_int, label_mapping


def apply_label_mapping(newick, label_mapping):
    """
    Apply an integer-taxon label mapping (label_mapping)
    from a string-based newick (where leaves are strings)
    and produce a mapped integer-based newick (where leaves are integers)

    Parameters
    ----------
    newick : str
        Newick with integer labels
    label_mapping : Dict[int, str]
        Mapping of leaf labels (as integers) to taxa

    Returns
    -------
    newick : str
        Newick with string labels
    """
    newick_str = core.apply_label_mapping(newick, label_mapping)

    return newick_str
