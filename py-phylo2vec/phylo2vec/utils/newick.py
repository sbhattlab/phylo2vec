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

    Examples
    --------
    >>> from phylo2vec.utils.newick import find_num_leaves
    >>> find_num_leaves("(((0,(3,5)6)8,2)9,(1,4)7)10;")
    6
    """
    return core.find_num_leaves(newick)


def remove_branch_lengths(newick: str) -> str:
    """Remove branch lengths annotations from a Newick string

    Example: "(((2:0.02,1:0.01),0:0.041),3:1.42);" --> "(((2,1),0),3);"

    Parameters
    ----------
    newick : str
        Newick string

    Returns
    -------
    str
        Newick string without branch lengths

    Examples
    --------
    >>> from phylo2vec.utils.newick import remove_branch_lengths
    >>> remove_branch_lengths("(((2:0.02,1:0.01),0:0.041),3:1.42);")
    '(((2,1),0),3);'
    """
    return core.remove_branch_lengths(newick)


def remove_parent_labels(newick: str) -> str:
    """Remove parent labels from the Newick string

    Example: "(((2,1)4,0)5,3)6;;" --> "(((2,1),0),3);"

    Parameters
    ----------
    newick : str
        Newick representation of a tree

    Returns
    -------
    str
        Newick string without parental/internal labels

    Examples
    --------
    >>> from phylo2vec.utils.newick import remove_parent_labels
    >>> remove_parent_labels("(((0,(3,5)6)8,2)9,(1,4)7)10;")
    '(((0,(3,5)),2),(1,4));'
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

    Examples
    --------
    >>> from phylo2vec.utils.newick import create_label_mapping
    >>> newick_int, mapping = create_label_mapping("((cat,dog),(fish,bird));")
    >>> isinstance(mapping, dict)
    True
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

    Examples
    --------
    >>> from phylo2vec.utils.newick import create_label_mapping, apply_label_mapping
    >>> newick_int, mapping = create_label_mapping("((cat,dog),(fish,bird));")
    >>> newick_str = apply_label_mapping(newick_int, mapping)
    >>> "cat" in newick_str and "dog" in newick_str
    True
    """
    newick_str = core.apply_label_mapping(newick, label_mapping)

    return newick_str
