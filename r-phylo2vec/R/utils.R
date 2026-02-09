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
