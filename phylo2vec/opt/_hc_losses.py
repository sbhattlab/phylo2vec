"""Loss functions for hill-climbing optimisation."""
import os
import re
import subprocess
import sys

from pathlib import PurePosixPath

from phylo2vec.base import to_newick

# Regex for a negative float
NEG_FLOAT_PATTERN = re.compile(r"-\d+.\d+")

# Test if the current platform is Windows or not
IS_WINDOWS = sys.platform.startswith("win")


def raxml_loss(
    v,
    taxa_dict,
    fasta_path,
    tree_folder_path,
    substitution_model,
    outfile="tmp.tree",
    **kwargs,
):
    """Compute loss for a given v via RaXML-NG.

    Parameters
    ----------
    v : numpy.ndarray or list
        v representation of a tree
    taxa_dict : dict[int, str]
        Current mapping of leaf labels (integer) to taxa
    fasta_path : str
        Path to fasta file
    tree_folder_path : str
        Path to a folder which will contain all intermediary and best trees
    substitution_model : str
        DNA/AA substitution model
    outfile : str, optional
        Path to a temporary tree written in Newick format, by default 'tmp.tree'

    Returns
    -------
    float
        Negative log-likelihood computed using RaXML-NG
    """
    try:
        newick = to_newick(v)
    except Exception as err:
        raise ValueError(f"Error for v = {repr(v)}") from err

    for taxa_key, taxa_name in taxa_dict.items():
        newick = re.sub(rf"([^\d]){taxa_key}([,)])", rf"\1{taxa_name}\2", newick)

    with open(
        os.path.join(tree_folder_path, outfile), "w", encoding="utf-8"
    ) as nw_file:
        nw_file.write(newick)

    return exec_raxml_ng(
        fasta_path=str(PurePosixPath(fasta_path.replace("C:", "/mnt/c"))),
        tree_path=str(
            PurePosixPath(tree_folder_path.replace("C:", "/mnt/c/"), outfile)
        ),
        substitution_model=substitution_model,
        **kwargs,
    )


def exec_raxml_ng(
    fasta_path, tree_path, substitution_model, cmd="raxml-ng", no_files=True
):
    """Optimize branch lengths and free model parameters on a fixed topology
    using RaxML-NG (https://github.com/amkozlov/raxml-ng)

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file (MSA)
    tree_path : str
        Path to tree file (Newick representation of the tree)
    substitution_model : str
        DNA evolution model
    cmd : str, optional
        Location of the RAxML-nG executable, by default "raxml-ng"
    no_files : bool, optional
        If True, add the "nofiles" option to raxml

    Returns
    -------
    float
        Negative log-likelihood after optimization
    """
    commands = [
        cmd,
        "--evaluate",
        "--msa",
        fasta_path,
        "--tree",
        tree_path,
        "--model",
        substitution_model,
        "--brlen",
        "scaled",
        "--log",
        "RESULT",
        "--threads",
        "1",
    ]

    if no_files:
        commands.append("--nofiles")

    if IS_WINDOWS:
        commands.insert(0, "wsl")  # Use Windows Subsystem for Linux
    else:
        commands = " ".join(commands)  # For Linux

    try:
        output = subprocess.run(
            commands, capture_output=True, check=True, shell=not IS_WINDOWS
        )
    except subprocess.CalledProcessError as _:
        # pylint: disable=subprocess-run-check
        output = subprocess.run(commands, capture_output=True, shell=not IS_WINDOWS)
        # pylint: enable=subprocess-run-check

        raise RuntimeError(output) from _

    stdout = output.stdout.decode("ascii")

    lik_line = [
        line for line in stdout.split("\n") if line.startswith("Final LogLikelihood")
    ][0]

    nll = -1 * float(re.findall(NEG_FLOAT_PATTERN, lik_line)[0])

    return nll
