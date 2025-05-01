# File to run binary build and install for the package
# Usage: Rscript install-package.R <package-path>

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  # default path from root
  args[1] <- paste(getwd(), "r-phylo2vec", sep = "/")
}

devtools::document(args[1])

fn <- devtools::build(args[1], binary = TRUE, args = c("--preclean"))

devtools::install_local(fn, force = TRUE)

if (file.exists(fn)) {
  file.remove(fn)
}
