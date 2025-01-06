library(testthat)
library(phylo2vec)

test_that(desc = "Vector to newick", code = {

  vec <- c(0L, 0L, 0L, 1L, 3L)

  newick <- to_newick(vec)

  # Test that the result is the correct value
  expect_equal( newick, "(((0,(3,5)6)8,2)9,(1,4)7)10;");
})
