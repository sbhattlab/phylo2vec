source("config.R")

library(ape)

test_that(desc = "cophenetic_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, 50)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)
      newick <- to_newick_from_vector(v)

      # Convert to Newick and back
      coph_p2v <- cophenetic_distances(v)
      tr <- read.tree(text = newick)
      tr <- compute.brlen(tr, 1)
      coph_ape <- cophenetic(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_equal(coph_p2v, coph_ape)
    }
  }
})
