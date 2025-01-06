# File to run tests for the package
# Usage: Rscript run-tests.R <package-path>

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  # default path from root
  args[1] = "./r-phylo2vec"
}

devtools::test(args[1], stop_on_failure = TRUE)
