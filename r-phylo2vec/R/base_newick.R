#' Convert a Phylo2Vec vector or matrix to Newick format
#'
#' @param vector_or_matrix Phylo2Vec matrix if branch lengths
#' are present, otherwise a vector
#' @return Newick string representation of the tree
#' @export
to_newick <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    # If the input is a vector, call the C function for vector
    .Call(wrap__to_newick_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    # If the input is a matrix, call the C function for matrix
    .Call(wrap__to_newick_from_matrix, vector_or_matrix)
  } else {
    stop("Input must be either a vector or a 2D matrix.")
  }
}

#' Convert a Newick string to a Phylo2Vec vector or matrix
#'
#' @param newick Newick string representation of the tree
#' @return A Phylo2Vec matrix (if branch lengths are present),
#' otherwise a Phylo2Vec vector (no branch lengths)
#' @export
from_newick <- function(newick) {
  if (.Call(wrap__has_branch_lengths, newick)) {
    .Call(wrap__to_matrix, newick)
  } else {
    .Call(wrap__to_vector, newick)
  }
}
