library(tools)


ACCEPTED_FILE_EXTENSIONS <- list(
  array = c("csv", "txt"),
  newick = c("txt", "nwk", "newick", "tree", "treefile")
)

check_path <- function(filepath, filetype) {
  suffix <- file_ext(filepath)

  if (!suffix %in% ACCEPTED_FILE_EXTENSIONS[[filetype]]) {
    stop(
      paste0(
        "Unsupported file extension: ",
        suffix, ". Accepted extensions: ",
        paste(ACCEPTED_FILE_EXTENSIONS[[filetype]], collapse = ", "),
        "."
      )
    )
  }
}

#' Read a text/csv file into a Phylo2Vec vector or matrix.
#'
#' @param filepath File path to read.
#' @param delimiter Character used to separate values, defaults to ",".
#' @return A vector (ndim == 1) or matrix (ndim == 2).
#'  which satisfies Phylo2Vec constraints
#' @export
load_p2v <- function(filepath, delimiter = ",") {
  check_path(filepath, "array")

  arr <- as.matrix(read.table(filepath, header = FALSE, sep = delimiter))

  if (dim(arr)[2] == 1) {
    # Convert to a vector
    arr <- as.numeric(arr)
  } else if (dim(arr)[2] == 3) {
    # Remove dimnames to preserve the original structure
    dimnames(arr) <- NULL
  } else {
    stop(
      paste0(
        "Input file should either be a vector (n_leaves x 1)",
        " or a matrix (n_leaves x 3)"
      )
    )
  }

  arr
}

#' Read a newick file into a Phylo2Vec vector or matrix.
#'
#' @param filepath_or_buffer File path or string containing
#' a Newick-formatted tree
#' @return A list containing:
#'  - newick: Newick string with integer labels
#'  - mapping: A named vector mapping original labels to integers
#'  - p2v: A phylo2ec representation, either a
#'    vector (ndim == 1) or matrix (ndim == 2)
#'    which satisfies phylo2vec constraints
#' @export
load_newick <- function(filepath_or_buffer) {
  if (file.exists(filepath_or_buffer)) {
    check_path(filepath_or_buffer, "newick")
    newick <- readLines(filepath_or_buffer)
  } else {
    newick <- filepath_or_buffer
  }

  res <- process_newick(newick)

  res$p2v <- from_newick(res$newick)

  res
}

#' Save a Phylo2Vec vector or matrix to Newick format into a file.
#'
#' @param vector_or_matrix A vector (ndim == 1) or matrix (ndim == 2)
#' @param filepath File path to save the Newick-formatted tree
#' @param labels Optional labels for the leaves of the tree
#' @export
save_newick <- function(vector_or_matrix, filepath, labels = NULL) {
  check_path(filepath, "newick")

  newick <- to_newick(vector_or_matrix)

  if (!is.null(labels)) {
    newick <- apply_label_mapping(newick, labels)
  }

  writeLines(newick, filepath)
}

#' Save a Phylo2Vec vector or matrix to a file.
#'
#' @param vector_or_matrix A vector (ndim == 1) or matrix (ndim == 2)
#'  which satisfies Phylo2Vec constraints
#' @param filepath Path to the output file
#' @param delimiter Character used to separate values, defaults to ",".
save_p2v <- function(vector_or_matrix, filepath, delimiter = ",") {
  check_path(filepath, "array")

  if (!(is.vector(vector_or_matrix) || is.matrix(vector_or_matrix))) {
    stop(
      paste0(
        "Input should either be a vector (ndim == 1) ",
        "or matrix (ndim == 2)"
      )
    )
  }

  write.table(
    vector_or_matrix,
    filepath,
    sep = delimiter,
    col.names = FALSE,
    row.names = FALSE
  )
}
