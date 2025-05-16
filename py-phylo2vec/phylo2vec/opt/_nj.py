# from ._base import BaseOptimizer


# class NeighborJoiningOptimizer(BaseOptimizer):
#     """
#     Neighbor Joining algorithm for phylogenetic tree construction.
#     """

#     def __init__(self, distance_matrix, random_seed=None):
#         super().__init__(random_seed)
#         self.distance_matrix = distance_matrix
#         self.tree = None

#     def _get_distance_matrix_triu(self, fasta_path):
#         raise NotImplementedError

#     def _optimise(self, fasta_path, v, label_mapping):
#         dm = self._get_distance_matrix_triu(fasta_path)

#         dm_min = dm.min()

#         next_parent = len(v) + 1
