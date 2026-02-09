#' Compute the precision matrix of a phylo2vec
#' vector (tree topology) or matrix (topology + branch lengths).
#'
#' The precision matrix is the inverse of the variance-covariance matrix.
#'
#' Output shape: (n_leaves, n_leaves)
#'
#' @param vector_or_matrix phylo2vec vector (ndim == 1)/matrix (ndim == 2)
#' @return Precision matrix
#' @export
precision <- function(vector_or_matrix) {
  precursor <- pre_precision(vector_or_matrix)

  # nrow(precursor) = ncol(precursor) = 2*k where k = n_leaves - 1
  n <- nrow(precursor)

  if (n <= 2) {
    return(precursor)
  }

  n_leaves <- nrow(precursor) / 2 + 1
  # Schur complement of the precursor matrix
  a <- precursor[1:n_leaves, 1:n_leaves]
  b <- precursor[1:n_leaves, (n_leaves + 1):n]
  c <- precursor[(n_leaves + 1):n, (n_leaves + 1):n]
  d <- precursor[(n_leaves + 1):n, 1:n_leaves]
  a - b %*% solve(c, d)
}

#' Compute the cophenetic distance matrix of a phylo2vec tree.
#'
#' The cophenetic distance between two leaves is the distance from each leaf
#' to their most recent common ancestor.
#' For vectors, this is the topological distance.
#' For matrices, this uses branch lengths.
#'
#' @param tree phylo2vec vector (1D) or matrix (2D)
#' @param unrooted If TRUE, compute unrooted distances. Default is FALSE.
#' @return Cophenetic distance matrix (shape: (n_leaves, n_leaves))
#' @export
cophenetic_distances <- function(tree, unrooted = FALSE) {
  .Call(wrap__cophenetic_distances, tree, unrooted)
}

#' Compute the Robinson-Foulds distance between two trees.
#'
#' RF distance counts the number of bipartitions (splits) that differ
#' between two tree topologies. Lower values indicate more similar trees.
#'
#' @param tree1 First tree as phylo2vec vector (1D) or matrix (2D).
#'   Only topology is used; branch lengths are ignored.
#' @param tree2 Second tree as phylo2vec vector (1D) or matrix (2D).
#'   Only topology is used; branch lengths are ignored.
#' @param normalize If TRUE, return normalized distance in range `[0.0, 1.0]`.
#'   Default is FALSE.
#' @return RF distance (numeric). Integer value if normalize=FALSE,
#'   float in `[0, 1]` otherwise.
#' @examples
#' v1 <- sample_tree(10, topology_only = TRUE)
#' v2 <- sample_tree(10, topology_only = TRUE)
#' robinson_foulds(v1, v1) # Identical trees: 0
#' robinson_foulds(v1, v2) # Different trees: >= 0
#' robinson_foulds(v1, v2, normalize = TRUE) # Normalized: [0, 1]
#' @export
robinson_foulds <- function(tree1, tree2, normalize = FALSE) {
  # Extract topology (first column) if matrix input
  v1 <- if (is.matrix(tree1)) as.integer(tree1[, 1]) else tree1
  v2 <- if (is.matrix(tree2)) as.integer(tree2[, 1]) else tree2

  .Call(wrap__robinson_foulds, v1, v2, normalize)
}

#' Compute the incidence matrix of a phylo2vec vector (tree topology).
#'
#' The incidence matrix B_ij is:
#' - 1 if edge_j leaves node_i
#' - -1 if edge_j enters node_i
#' - 0 otherwise
#'
#' Output shape: (2 * k + 1, 2 * k) where k = n_leaves - 1
#'
#' @param vector_or_matrix phylo2vec vector (ndim == 1)
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
      return(Matrix::sparseMatrix(
        i = coo$rows,
        j = coo$cols,
        x = coo$data,
        index1 = FALSE,
        dims = dims
      ))
    } else if (format == "csr" || format == "R") {
      csr <- .Call(wrap__incidence_csr, vector)
      return(Matrix::sparseMatrix(
        j = csr$indices,
        p = csr$indptr,
        x = csr$data,
        index1 = FALSE,
        dims = dims
      ))
    } else if (format == "csc" || format == "C") {
      csc <- .Call(wrap__incidence_csc, vector)
      return(Matrix::sparseMatrix(
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
    stop("Input must be a phylo2vec vector.")
  }
}
