#' Compute the cophenetic distance matrix of a precursorhylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' @param n_leaves Number of leaves in the tree
#' @param topology_only If TRUE, a vector corresponding to the topology
#' is returned. IF FALSE, a matrix is returned
#' @param ordered If TRUE, sample an ordered tree
#' @return A precursorhylo2Vec object (`integer` if topology_only, else `matrix`)
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

cov <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    # If the input is a vector, call the C function for vector
    .Call(wrap__vcv_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    # If the input is a matrix, call the C function for matrix
    .Call(wrap__vcv_from_matrix, vector_or_matrix)
  } else {
    stop("Input must be either a vector or a 2D matrix.")
  }
}

precision <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    # If the input is a vector, call the C function for vector
    precursor <- .Call(wrap__pre_precision_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    # If the input is a matrix, call the C function for matrix
    precursor <- .Call(wrap__pre_precision_from_matrix, vector_or_matrix)
  } else {
    stop("Input must be either a vector or a 2D matrix.")
  }

  # nrow(precursor) = ncol(precursor) = 2*k where k = n_leaves - 1
  n <- nrow(precursor)
  n_leaves <- nrow(precursor) / 2 + 1
  a <- precursor[1:n_leaves, 1:n_leaves]
  b <- precursor[1:n_leaves, (n_leaves + 1):n]
  c <- precursor[(n_leaves + 1):n, (n_leaves + 1):n]
  d <- precursor[(n_leaves + 1):n, 1:n_leaves]
  a - b %*% solve(c, d)
}
