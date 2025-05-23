#' Compute the cophenetic distance matrix of a Phylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' @param n_leaves Number of leaves in the tree
#' @param topology_only If TRUE, a vector corresponding to the topology
#' is returned. IF FALSE, a matrix is returned
#' @param ordered If TRUE, sample an ordered tree
#' @return A Phylo2Vec object (`integer` if topology_only, else `matrix`)
#' @export
cophenetic_distances <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    # If the input is a vector, call the C function for vector
    .Call(wrap__cophenetic_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    # If the input is a matrix, call the C function for matrix
    .Call(wrap__cophenetic_from_matrix, vector_or_matrix)
  } else {
    stop("Input must be either a vector or a 2D matrix.")
  }
}
