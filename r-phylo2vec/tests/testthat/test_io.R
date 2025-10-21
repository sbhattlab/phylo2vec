source("config.R")

test_save_and_load <- function(topology_only) {
  tmp_path <- tempdir()
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    vector_or_matrix <- sample_tree(
      n_leaves,
      ordered = FALSE,
      topology_only = topology_only
    )

    # Save and load via CSV
    # Check that the loaded data matches the original
    csv_path <- file.path(tmp_path, "test.csv")
    save_p2v(vector_or_matrix, csv_path)
    vector_or_matrix2 <- load_p2v(csv_path)
    expect_equal(vector_or_matrix, vector_or_matrix2)

    # Unallowed file extension should trigger an error
    random_path <- file.path(tmp_path, "test.random")
    expect_error(save_p2v(vector_or_matrix, random_path))
  }
}

test_that(desc = "save_and_load_vector", {
  test_save_and_load(topology_only = TRUE)
})

test_that(desc = "save_and_load_matrix", {
  test_save_and_load(topology_only = FALSE)
})

test_save_and_load_newick <- function(topology_only) {
  tmp_path <- tempdir()
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    vector_or_matrix <- sample_tree(
      n_leaves,
      ordered = FALSE,
      topology_only = topology_only
    )

    # Save and load via Newick
    # Check that the loaded data matches the original
    newick_path <- file.path(tmp_path, "test.newick")
    save_newick(vector_or_matrix, newick_path)
    vector_or_matrix2 <- load_newick(newick_path)
    expect_equal(vector_or_matrix, vector_or_matrix2)

    # Unallowed file extension should trigger an error
    random_path <- file.path(tmp_path, "test.random")
    expect_error(save_newick(vector_or_matrix, random_path))
  }
}

test_that(desc = "save_and_load_newick_vector", {
  test_save_and_load_newick(topology_only = TRUE)
})

test_that(desc = "save_and_load_newick_matrix", {
  test_save_and_load_newick(topology_only = FALSE)
})

test_save_newick_with_labels <- function(...) {
  tmp_path <- tempdir()
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    tr <- ape::rtree(n_leaves, rooted = TRUE, equiprob = TRUE, ...)

    nw_str <- ape::write.tree(tr)

    out <- create_label_mapping(nw_str)

    vector_or_matrix <- from_newick(out$newick)

    newick_path <- file.path(tmp_path, "test.newick")
    save_newick(vector_or_matrix, newick_path, labels = out$mapping)

    nw_str2 <- readLines(newick_path, warn = FALSE)

    expect_equal(nw_str, nw_str2)
  }
}

test_that(desc = "save_newick_with_labels_vector", {
  test_save_newick_with_labels(br = NULL)
})

test_that(desc = "save_newick_with_labels_matrix", {
  test_save_newick_with_labels()
})

test_load_newick_from_str <- function(topology_only) {
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    vector_or_matrix <- sample_tree(
      n_leaves,
      ordered = FALSE,
      topology_only = topology_only
    )
    nw_str <- to_newick(vector_or_matrix)
    vector_or_matrix2 <- load_newick(nw_str)
    expect_equal(vector_or_matrix, vector_or_matrix2)
  }
}

test_that(desc = "load_newick_from_str_vector", {
  test_load_newick_from_str(topology_only = TRUE)
})

test_that(desc = "load_newick_from_str_matrix", {
  test_load_newick_from_str(topology_only = FALSE)
})

test_that(desc = "load_nbyNot3", {
  tmp_path <- tempdir()
  csv_path <- file.path(tmp_path, "test.csv")

  # N x 2
  writeLines("0,0.1\n1,0.2", csv_path)
  expect_error(load_p2v(csv_path))

  # N x 4
  writeLines("0,0.1,0,0.5\n1,0.2,1,0.6", csv_path)
  expect_error(load_p2v(csv_path))
})


test_that(desc = "save_empty_vector", {
  tmp_path <- tempdir()
  csv_path <- file.path(tmp_path, "test.csv")

  # Triggers the stop arm of save_p2v
  expect_error(save_p2v(c(), csv_path))
})

test_that(desc = "save_empty_matrix", {
  tmp_path <- tempdir()
  csv_path <- file.path(tmp_path, "test.csv")

  # Triggers the stop arm of save_p2v
  expect_error(save_p2v(matrix(array()), csv_path))

  # NOTE: This triggers a different error: panic error in check_m
  # expect_error(save_p2v(matrix(array(0)), csv_path))
})

test_that(desc = "save_ndim3", {
  tmp_path <- tempdir()
  csv_path <- file.path(tmp_path, "test.csv")

  x <- array(runif(60), dim = c(3, 4, 5))

  expect_error(save_p2v(x, csv_path))
})
