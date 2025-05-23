"""File validation functions for Phylo2Vec I/O."""

from pathlib import Path


FILE_EXTENSIONS = {
    "array": [".csv", ".txt"],
    "newick": [".txt", ".nwk", ".newick", ".tree", ".treefile"],
}


def check_path(filepath: str, file_type: str):
    suffix = Path(filepath).suffix
    file_extensions = FILE_EXTENSIONS[file_type]
    assert suffix in file_extensions, (
        f"Unsupported file extension: {suffix}. "
        f"Accepted extensions for {file_type} files: {file_extensions}"
    )
