library(testthat)
library(phylo2vec)

# Minimum number of leaves
MIN_N_LEAVES <- 5

# Maximum number of leaves
MAX_N_LEAVES <- 200

# Number of test repeats for a tree with n leaves
N_REPEATS <- 7

# Random seed for reproducibility
RANDOM_SEED <- 42

set.seed(RANDOM_SEED)
