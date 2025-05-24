# File to run binary build and install for the package
# Usage: Rscript install-package.R <package-path>

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  # default path from root
  args[1] <- paste(getwd(), "r-phylo2vec", sep = "/")
}

devtools::document(args[1])

devtools::install(args[1], build = FALSE)
