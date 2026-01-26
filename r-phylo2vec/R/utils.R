#' Sample a random tree as a phylo2vec object
#'
#' @param n_leaves Number of leaves in the tree
#' @param topology_only If TRUE, a vector corresponding to the topology
#' is returned. IF FALSE, a matrix is returned
#' @param ordered If TRUE, sample an ordered tree
#' @return A phylo2vec object (`integer` if topology_only, else `matrix`)
#' @export
sample_tree <- function(n_leaves, topology_only = FALSE, ordered = FALSE) {
  if (topology_only) {
    # Sample a random topology
    .Call(wrap__sample_vector, n_leaves, ordered)
  } else {
    # Sample a random tree with branch lengths
    .Call(wrap__sample_matrix, n_leaves, ordered)
  }
}

#' Get the most recent common ancestor between two nodes in a phylo2vec tree.
#'
#' `node1` and `node2` can be leaf nodes (0 to n_leaves - 1)
#' or internal nodes (n_leaves to 2 * n_leaves - 2).
#'
#' Similar to ape's `getMRCA` function (for leaf nodes)
#' and ETE's `get_common_ancestor` (for all nodes), but for phylo2vec.
#'
#' Note: The MRCA is purely topological, so for matrices, only the vector
#' component (column 1) is used.
#'
#' @param vector_or_matrix phylo2vec vector (integer) or matrix (numeric)
#' @param node1 A node in the tree (0-indexed)
#' @param node2 A node in the tree (0-indexed)
#' @return The most recent common ancestor node (0-indexed)
#' @export
get_common_ancestor <- function(vector_or_matrix, node1, node2) {
  if (is.vector(vector_or_matrix, "integer")) {
    get_common_ancestor_from_vector(vector_or_matrix, node1, node2)
  } else if (is.matrix(vector_or_matrix)) {
    v <- as.integer(vector_or_matrix[, 1])
    get_common_ancestor_from_vector(v, node1, node2)
  } else {
    stop("Input must be either an integer vector or a 2D matrix.")
  }
}

#' Get the depth of a node in a phylo2vec tree.
#'
#' The depth of a node is the length of the path from the root to that node
#' (i.e., distance from root). This follows the BEAST/ETE convention.
#'
#' The root has depth 0, and depths increase as you move toward the leaves.
#'
#' For vectors, topological depth is returned (all branch lengths = 1).
#' For matrices, actual branch lengths are used.
#'
#' @param vector_or_matrix phylo2vec vector (ndim == 1)/matrix (ndim == 2)
#' @param node A node in the tree (0-indexed, from 0 to 2*n_leaves - 2)
#' @return Depth of the node (distance from root)
#' @export
get_node_depth <- function(vector_or_matrix, node) {
  if (is.vector(vector_or_matrix, "integer")) {
    get_node_depth_from_vector(vector_or_matrix, node)
  } else if (is.matrix(vector_or_matrix)) {
    get_node_depth_from_matrix(vector_or_matrix, node)
  } else {
    stop("Input must be either an integer vector or a 2D matrix.")
  }
}

#' Get the depths of all nodes in a phylo2vec tree.
#'
#' The depth of a node is the length of the path from the root to that node
#' (i.e., distance from root). This follows the BEAST/ETE convention.
#'
#' The root has depth 0, and depths increase as you move toward the leaves.
#'
#' For vectors, topological depth is returned (all branch lengths = 1).
#' For matrices, actual branch lengths are used.
#'
#' @param vector_or_matrix phylo2vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Vector of depths for all nodes (length = 2 * n_leaves - 1).
#'   Index i+1 contains the depth of node i (R is 1-indexed).
#' @export
get_node_depths <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix, "integer")) {
    get_node_depths_from_vector(vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    get_node_depths_from_matrix(vector_or_matrix)
  } else {
    stop("Input must be either an integer vector or a 2D matrix.")
  }
}
