API Reference
=============

.. currentmodule:: phylo2vec

Base
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    from_ancestry
    from_edges
    from_newick
    from_pairs
    to_ancestry
    to_edges
    to_newick
    to_pairs

IO
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    load
    load_newick
    save
    save_newick


Statistics
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    stats.b2
    stats.cophenetic_distances
    stats.cov
    stats.incidence
    stats.leaf_depth_variance
    stats.pairwise_distances
    stats.precision
    stats.robinson_foulds
    stats.sackin

Utils
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    sample_matrix
    sample_vector
    utils.matrix.check_matrix
    utils.newick.apply_label_mapping
    utils.newick.create_label_mapping
    utils.newick.find_num_leaves
    utils.newick.remove_branch_lengths
    utils.newick.remove_parent_labels
    utils.vector.add_leaf
    utils.vector.check_vector
    utils.vector.get_common_ancestor
    utils.vector.get_node_depth
    utils.vector.get_node_depths
    utils.vector.queue_shuffle
    utils.vector.remove_leaf
    utils.vector.reroot
    utils.vector.reroot_at_random

Datasets
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    datasets.list_datasets
    datasets.load_alignment
    datasets.load_descr
    datasets.load_fasta
    datasets.read_fasta

Optimisation
----------------

Requires the optional dependencies: ``pip install phylo2vec[opt]``

.. autosummary::
    :nosignatures:
    :toctree: generated/

    opt.GradME
    opt.HillClimbing
    opt.gradme_loss
    opt.list_methods
