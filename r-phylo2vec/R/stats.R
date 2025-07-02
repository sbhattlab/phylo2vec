#' Compute the cophenetic distance matrix of a Phylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' @param vector_or_matrix Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Cophenetic distance matrix
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

#' Compute the variance-covariance matrix of a Phylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' @param vector_or_matrix Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Variance-covariance matrix
#' @export
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

#' Compute the precision matrix of a Phylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' The precision matrix is the inverse of the variance-covariance matrix.
#'
#' @param vector_or_matrix Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Precision matrix
#' @export
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

  # Schur complement of the precursor matrix
  # nrow(precursor) = ncol(precursor) = 2*k where k = n_leaves - 1
  n <- nrow(precursor)
  n_leaves <- nrow(precursor) / 2 + 1
  a <- precursor[1:n_leaves, 1:n_leaves]
  b <- precursor[1:n_leaves, (n_leaves + 1):n]
  c <- precursor[(n_leaves + 1):n, (n_leaves + 1):n]
  d <- precursor[(n_leaves + 1):n, 1:n_leaves]
  a - b %*% solve(c, d)
}
