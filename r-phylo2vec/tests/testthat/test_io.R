source("config.R")

test_that(desc = "save_and_load_vector", {
  tmp_path <- tempdir()
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      v <- sample_vector(n_leaves, FALSE)

      csv_path <- file.path(tmp_path, "test.csv")
      save_p2v(v, csv_path)
      v2 <- load_p2v(csv_path)
      expect_equal(v, v2)

      random_path <- file.path(tmp_path, "test.random")
      expect_error(save_p2v(v, random_path))
    }
  }
})

test_that(desc = "save_and_load_matrix", {
  tmp_path <- tempdir()
  for (n_leaves in c(2, MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      m <- sample_matrix(n_leaves, FALSE)

      csv_path <- file.path(tmp_path, "test.csv")
      save_p2v(m, csv_path)
      m2 <- load_p2v(csv_path)
      expect_equal(m, m2)
    }
  }
})
