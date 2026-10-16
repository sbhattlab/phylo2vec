import uuid

from importlib import resources

from phylo2vec.datasets import list_datasets, load_alignment
from phylo2vec.datasets._base import DATA_MODULE


def test_load_available_datasets():
    """Test loading all available datasets"""
    datasets = list_datasets()
    for dataset in datasets:
        records, descr = load_alignment(dataset)
        record = next(records)
        seqlen = len(record.seq)
        alncount = sum(1 for _ in records) + 1
        assert seqlen == descr["Number of Bases"]
        assert alncount == descr["Number of Taxa"]


def _write_fasta(path, n_taxa):
    path.write_text("".join(f">taxon{i}\nACGT\n" for i in range(n_taxa)))


def test_load_alignment_from_url(tmp_path):
    """Test loading a custom dataset from a URL (file:// avoids network access)"""
    src = tmp_path / f"alignment_{uuid.uuid4().hex}.fa"
    _write_fasta(src, 3)
    # Downloaded files are stored in the package data
    stored = resources.files(DATA_MODULE).joinpath(src.name)
    try:
        records, descr = load_alignment(src.as_uri(), data_format="fasta")
        assert descr is None
        assert [record.id for record in records] == ["taxon0", "taxon1", "taxon2"]

        # The stored file is reused unless overwrite=True
        _write_fasta(src, 4)
        records, _ = load_alignment(src.as_uri(), data_format="fasta")
        assert sum(1 for _ in records) == 3
        records, _ = load_alignment(src.as_uri(), data_format="fasta", overwrite=True)
        assert sum(1 for _ in records) == 4
    finally:
        if stored.is_file():
            stored.unlink()
