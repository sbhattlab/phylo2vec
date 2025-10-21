source("config.R")

test_that(desc = "sample_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Check that the vector is valid
      expect_no_error(check_v(v))
    }
  }
})

test_that(desc = "sample_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Check that the matrix is valid
      expect_no_error(check_m(m))
    }
  }
})

test_that(desc = "find_num_leaves", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to Newick
      newick <- to_newick(v)

      # Find number of leaves
      n_leaves_found <- find_num_leaves(newick)

      expect_equal(n_leaves, n_leaves_found)
    }
  }
})

test_that(desc = "remove_and_add", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      leaf <- sample.int(n_leaves, 1) - 1

      v_and_sis <- remove_leaf(v, leaf)

      v_sub <- v_and_sis$v
      sis <- v_and_sis$branch

      if (sis >= leaf) {
        sis <- sis - 1
      }

      v_add <- add_leaf(v_sub, leaf, sis)

      expect_equal(v, v_add)
    }
  }
})

test_that(desc = "get_common_ancestor", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Generate two random leaf nodes
      nodes <- sample.int(n_leaves, 2)
      node1 <- nodes[1] - 1
      node2 <- nodes[2] - 1

      # Get the common ancestor using the Phylo2Vec function
      ca <- get_common_ancestor(v, node1, node2)

      # Get the common ancestor using ape
      tr <- ape::read.tree(text = to_newick(v))

      ca_ape <- ape::getMRCA(tr, c(as.character(node1), as.character(node2)))

      ca_ape_converted <- as.integer(tr$node.label[[ca_ape - n_leaves]])

      expect_equal(ca, ca_ape_converted)
    }
  }
})

test_create_and_apply_label_mapping <- function(...) {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      tr <- ape::rtree(n_leaves, rooted = TRUE, equiprob = TRUE, ...)

      nw_str <- ape::write.tree(tr)

      out <- create_label_mapping(nw_str)

      nw_str_new <- apply_label_mapping(out$newick, out$mapping)

      expect_equal(nw_str, nw_str_new)
    }
  }
}

test_that(desc = "create_and_apply_label_mapping_vector", {
  test_create_and_apply_label_mapping(br = NULL)
})

test_that(desc = "create_and_apply_label_mapping_matrix", {
  test_create_and_apply_label_mapping()
})

test_that(desc = "remove_branch_lengths", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      m <- sample_matrix(n_leaves, FALSE)

      nw_from_m <- to_newick_from_matrix(m)

      nw_no_bl <- remove_branch_lengths(nw_from_m)

      v <- as.integer(m[, 1])

      nw_from_v <- to_newick_from_vector(v)

      expect_equal(nw_no_bl, nw_from_v)
    }
  }
})

test_that(desc = "has_branch_lengths", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    m <- sample_matrix(n_leaves, FALSE)

    nw_from_m <- to_newick_from_matrix(m)

    expect_true(has_branch_lengths(nw_from_m))

    v <- as.integer(m[, 1])

    nw_from_v <- to_newick_from_vector(v)

    expect_false(has_branch_lengths(nw_from_v))
  }
})
