"""Base IO code for all datasets."""

from importlib import resources

from Bio import SeqIO

DATA_MODULE = "phylo2vec.datasets.data"
DESCR_MODULE = "phylo2vec.datasets.descr"


def read_fasta(data_path):
    """Read a fasta file

    Parameters
    ----------
    data_path : str
        Path to fasta file

    Returns
    -------
    Generator
        An iterator for all sequences
        Each element has "id" (taxon name) and "seq" (sequence) attributes
    """
    return SeqIO.parse(data_path, "fasta")


def load_fasta(data_file_name, data_module=DATA_MODULE):
    """Loaded a preloaded fasta file

    Parameters
    ----------
    data_file_name : str
        Data file name
    data_module : str, optional
        Location of the data module, by default DATA_MODULE

    Returns
    -------
    Generator
        An iterator for all sequences
        Each element has "id" (taxon name) and "seq" (sequence) attributes
    """
    return read_fasta(_open_text(data_module, data_file_name))


def load_descr(descr_file_name, descr_module=DESCR_MODULE):
    """Loaded the description of a preloaded fasta file

    Parameters
    ----------
    dsecr_file_name : str
        Description file name
    data_module : str, optional
        Location of the data module, by default DATA_MODULE

    Returns
    -------
    str
        Contents of the description file
    """
    return _read_text(descr_module, descr_file_name)


# Largely inspired by https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/fixes.py
def _open_text(module, file_name, encoding="utf-8"):
    """Open for text reading the contents of resource within package as a str

    Parameters
    ----------
    module : str
        Location of the module
    file_name : str
        File name
    encoding : str, optional
        Encoding, by default "utf-8"

    Returns
    -------
    str
        Contents of the text file
    """
    return resources.files(module).joinpath(file_name).open("r", encoding=encoding)


# Largely inspired by https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/fixes.py
def _read_text(module, file_name, encoding="utf-8"):
    """Read and return the contents of resource within package as a str

    Parameters
    ----------
    module : str
        Location of the module
    file_name : str
        File name
    encoding : str, optional
        Encoding, by default "utf-8"

    Returns
    -------
    str
        Contents of the text file
    """
    return resources.files(module).joinpath(file_name).read_text(encoding=encoding)
