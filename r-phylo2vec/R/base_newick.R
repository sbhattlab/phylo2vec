#' Convert a Newick string to a phylo2vec vector or matrix
#'
#' @param newick Newick string representation of the tree
#' @return A phylo2vec matrix (tree with branch lengths)
#' or a phylo2vec vector (tree topology only)
#' @export
from_newick <- function(newick) {
  if (.Call(wrap__has_branch_lengths, newick)) {
    .Call(wrap__to_matrix, newick)
  } else {
    .Call(wrap__to_vector, newick)
  }
}
