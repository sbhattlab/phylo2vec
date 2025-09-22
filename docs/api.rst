API Reference
=============

.. currentmodule:: phylo2vec

Base
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    from_newick
    to_newick
    from_ancestry
    to_ancestry

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

    stats.cophenetic_distances
    stats.pairwise_distances
    stats.cov
    stats.precision
    stats.incidence

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
    utils.vector.check_vector
    utils.vector.add_leaf
    utils.vector.get_common_ancestor
    utils.vector.remove_leaf
