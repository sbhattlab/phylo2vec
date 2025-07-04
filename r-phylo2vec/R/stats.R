library(Matrix)

#' Compute the cophenetic distance matrix of a Phylo2Vec
#' vector (topological) or matrix (from branch lengths).
#'
#' @param vector_or_matrix Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Cophenetic distance matrix
#' @export
cophenetic_distances <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    .Call(wrap__cophenetic_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
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
vcovp <- function(vector_or_matrix) {
  if (is.vector(vector_or_matrix)) {
    .Call(wrap__vcov_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
    .Call(wrap__vcov_from_matrix, vector_or_matrix)
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
    precursor <- .Call(wrap__pre_precision_from_vector, vector_or_matrix)
  } else if (is.matrix(vector_or_matrix)) {
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

#' Compute the incidence matrix of a Phylo2Vec vector.
#'
#' @param vector_or_matrix Phylo2Vec vector (ndim == 1)
#' @param format The format of the incidence matrix.
#'   Options are "coo"/"T" (coordinate), "csr"/"R" (compressed sparse row),
#'   "csc"/"C" (compressed sparse column), "dense"/"D" (dense matrix).
#'   Default is "C" for compressed sparse row.
#' @return Incidence matrix in the specified format.
#' @export
incidence <- function(vector, format = "C") {
  if (is.vector(vector)) {
    k <- length(vector)
    dims <- c(2 * k + 1, 2 * k)
    if (format == "coo" || format == "T") {
      coo <- .Call(wrap__incidence_coo, vector)
      return(sparseMatrix(
        i = coo$rows,
        j = coo$cols,
        x = coo$data,
        index1 = FALSE,
        dims = dims
      ))
    } else if (format == "csr" || format == "R") {
      csr <- .Call(wrap__incidence_csr, vector)
      return(sparseMatrix(
        j = csr$indices,
        p = csr$indptr,
        x = csr$data,
        index1 = FALSE,
        dims = dims
      ))
    } else if (format == "csc" || format == "C") {
      csc <- .Call(wrap__incidence_csc, vector)
      return(sparseMatrix(
        i = csc$indices,
        p = csc$indptr,
        x = csc$data,
        index1 = FALSE,
        dims = dims
      ))
    } else if (format == "dense" || format == "D") {
      return(.Call(wrap__incidence_dense, vector))
    } else {
      stop("Unknown format. Use 'coo', 'csr', 'csc', or 'dense'.")
    }
  } else {
    stop("Input must be a Phylo2Vec vector.")
  }
}
