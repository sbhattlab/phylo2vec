library(testthat)
library(phylo2vec)

MIN_N_LEAVES <- 5
N_TESTS <- 200
MAX_N_LEAVES <- MIN_N_LEAVES + N_TESTS - 1
N_REPEATS <- 5
RANDOM_SEED <- 42

set.seed(RANDOM_SEED)
