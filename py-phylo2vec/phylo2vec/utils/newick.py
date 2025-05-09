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
    label_mapping : Dict str --> str
        Mapping of leaf labels (integers converted to string) to taxa
    """
    label_mapping = {}

    newick = newick[:-1]  # For ";"

    newick_int = newick

    def do_reduce(newick, newick_clean, j):
        for i, char in enumerate(newick):
            if char == "(":
                open_idx = i + 1
            elif char == ")":
                for child in newick[open_idx:i].split(",", 2):
                    if child not in label_mapping:
                        # Update the newick with an integer
                        newick_clean = newick_clean.replace(child, f"{j}")

                        # Add the taxon to the mapping
                        label_mapping[f"{j}"] = child
                        j += 1

                parent = newick[i + 1 :].split(",", 1)[0].split(")", 1)[0]

                newick = newick.replace(
                    newick[open_idx - 1 : i + 1 + len(parent)], f"{j - 1}"
                )

                newick_clean = newick_clean.replace(parent, "")

                return do_reduce(newick, newick_clean, j)

        return newick_clean

    newick_int = do_reduce(newick, newick_int, 0) + ";"

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
    label_mapping : Dict str --> str
        Mapping of leaf labels (integers converted to string) to taxa

    Returns
    -------
    newick : str
        Newick with string labels
    """
    for i in range(len(label_mapping)):
        key = f"{len(label_mapping) - i - 1}"

        newick = newick.replace(key, label_mapping[key])

    return newick
