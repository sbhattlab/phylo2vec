"""File validation functions for Phylo2Vec I/O."""

from pathlib import Path

ACCEPTED_ARRAY_FILE_EXTENSIONS = [".csv", ".txt"]
ACCEPTED_NEWICK_FILE_EXTENSIONS = [".txt", ".nwk", ".newick", ".tree", ".treefile"]


def check_array_path(filepath: str):
    """Check if the file path has a valid extension for Phylo2Vec array files.

    Throws an AssertionError if the file extension is not valid.

    Parameters
    ----------
    filepath : str
        Path to the input file
        Valid extensions are: .csv, .txt
    """
    suffix = Path(filepath).suffix
    assert suffix in ACCEPTED_ARRAY_FILE_EXTENSIONS, (
        f"Unsupported file extension: {suffix}. "
        f"Accepted extensions: {ACCEPTED_ARRAY_FILE_EXTENSIONS}"
    )


def check_newick_path(filepath: str):
    """Check if the file path has a valid extension for Newick files.

    Parameters
    ----------
    filepath : str
        Path to the input file
        Valid extensions are: .txt, .nwk, .newick, .tree, .treefile
    """
    suffix = Path(filepath).suffix
    assert suffix in ACCEPTED_NEWICK_FILE_EXTENSIONS, (
        f"Unsupported file extension: {suffix}. "
        f"Accepted extensions: {ACCEPTED_NEWICK_FILE_EXTENSIONS}"
    )
