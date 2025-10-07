"""Base IO code for all datasets."""

import json
import os
import re


from importlib import resources
from pathlib import Path
from urllib.parse import (
    urlparse,
    uses_netloc,
    uses_params,
    uses_relative,
)
from urllib.request import urlretrieve

from Bio import SeqIO

DATA_MODULE = "phylo2vec.datasets.data"
DESCR_MODULE = "phylo2vec.datasets.descr"
DATASETS = {
    "fluA": {
        "format": "fasta",
    },
    "h3n2": {
        "format": "fasta",
    },
    "m501": {
        "format": "fasta",
    },
    "yeast": {
        "format": "RData",
    },
    "zika": {
        "format": "fasta",
    },
}


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


FORMATS = {
    "fasta": {
        "ext": "fa",
        "loader": load_fasta,
    },
}


def list_datasets():
    """List all available datasets."""
    # Check that format is supported
    return [dataset for dataset in DATASETS if DATASETS[dataset]["format"] in FORMATS]


def is_url(maybe_url: str) -> bool:
    # pylint: disable=line-too-long
    """Check if a string is a URL.

    Adapted from pandas.
    Source: https://github.com/pandas-dev/pandas/blob/c8213d16d98079da2a7b7464b95fe22f7d72d427/pandas/io/common.py
    # Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
    # Copyright (c) 2011-2025, Open source contributors.
    # BSD 3-Clause License


    Parameters
    ----------
    maybe_url : str
        String to check
    Returns
    -------
    bool
        True if the string is a URL, False otherwise
    """
    # pylint: enable=line-too-long
    _VALID_URLS = set(uses_relative + uses_netloc + uses_params)
    _VALID_URLS.discard("")

    if not isinstance(maybe_url, str):
        return False
    return urlparse(maybe_url).scheme in _VALID_URLS


def load_alignment(
    dataset,
    data_format=None,
    data_module=DATA_MODULE,
    descr_module=DESCR_MODULE,
    url_key="Link",
    overwrite=False,
):
    if dataset in DATASETS:
        # Load description
        print(f"Loading {dataset}...")
        descr = load_descr(f"{dataset}.md", descr_module=descr_module)
        print(descr)
        print(json.dumps(descr, indent=2))
        # Load data format
        the_data_format = DATASETS[dataset]["format"]
        format_ext = FORMATS[the_data_format]["ext"]
        data_loader = FORMATS[the_data_format]["loader"]
        # Infer dataset path
        dataset_path = resources.files(data_module).joinpath(f"{dataset}.{format_ext}")

        if not os.path.exists(dataset_path) or overwrite:
            # Parse the url from the description file
            url_content = descr[url_key]
            if not is_url(url_content):
                # Try to extract URL from markdown link format [text](url)
                match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", url_content)
                url_path = match.group(2) if match else None
            else:
                url_path = url_content
            assert is_url(
                url_path
            ), f"Could not parse URL from description file for dataset {dataset}: {url_path}"
            urlretrieve(url_path, dataset_path)
    else:
        descr = None
        if data_format not in FORMATS:
            raise ValueError(
                (
                    "`data_format` was not provided or is not supported. "
                    "Are you trying to load a custom dataset? "
                    f"Supported formats: {list(FORMATS.keys())}"
                )
            )
        # Load data format
        the_data_format = FORMATS[data_format]
        format_ext = the_data_format["ext"]
        data_loader = the_data_format["loader"]

        if is_url(dataset):
            url_path = urlparse(dataset).path
            filename = f"{Path(url_path).stem}.{format_ext}"
            dataset_path = resources.files(data_module).joinpath(filename)

            if not dataset_path.exists():
                # Download the file
                urlretrieve(url_path, dataset_path)
            else:
                print(f"File {filename} already exists. Using the existing file.")
        elif os.path.exists(dataset):
            dataset_path = dataset
        else:
            raise ValueError(
                f"`dataset` {dataset} is not a valid dataset name, a valid path, or a valid URL."
            )

    # Load the data from saved file
    return data_loader(dataset_path), descr


def load_descr(descr_file_name, descr_module=DESCR_MODULE, header_line=2, data_line=4):
    """Loaded the description of a preloaded fasta file

    Parameters
    ----------
    dsecr_file_name : str
        Description file name
    data_module : str, optional
        Location of the data module, by default DATA_MODULE
    header_line : int, optional
        Line number of the header in the markdown table, by default 2
    data_line : int, optional
        Line number of the data in the markdown table, by default 4

    Returns
    -------
    str
        Contents of the description file
    """
    text = _read_text(descr_module, descr_file_name)
    # Parse markdown content to a dictionary
    lines = text.split("\n")
    # Header line
    keys = lines[header_line].split("|")[1:-1]
    # Data line
    values = lines[data_line].split("|")[1:-1]
    data = {}
    for key, val in zip(keys, values):
        key = key.strip()
        val = val.strip()
        if key in ("Number of Taxa", "Number of Bases"):
            data[key] = int(val)
        else:
            data[key] = val
    return data


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
