API Reference
=============

.. currentmodule:: phylo2vec

Base
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    base.to_newick
    base.to_vector

Metrics
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    metrics.cophenetic_distances

Utils
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    utils.find_num_leaves
    utils.create_label_mapping
    utils.apply_label_mapping
    utils.remove_branch_lengths
    utils.remove_parent_labels
    utils.sample_vector
    utils.seed_everything
    utils.check_v
    utils.reorder_v
    utils.reroot_at_random
    utils.remove_leaf
    utils.add_leaf
    utils.get_common_ancestor

Utils - IO
----------------
.. autosummary::
    :nosignatures:
    :toctree: generated/

    utils.read_newick_file
    utils.write_newick_file
    utils.read_vector_csv
    utils.write_vector_csv
    utils.read_newick_file_labeled
    utils.write_newick_file_labeled
