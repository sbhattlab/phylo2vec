source("config.R")

test_that(desc = "sample", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Check that the vector is valid
      expect_no_error(check_v(v))
    }
  }
})

test_that(desc = "remove_and_add", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(2)) {
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
