from phylo2vec.datasets import list_datasets, load_alignment


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
