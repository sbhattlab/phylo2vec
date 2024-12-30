# r-phylo2vec

**NOTE: This is currently in active development and APIs will change. Use at
your own risk.**

This directory contains the pylo2vec R codebase, which includes Rust binding
setup.

Open up `R` command line interactive mode within the pixi environment:

```console
pixi run -e r-phylo2vec R --interactive
```

Once in R, you can run the following to start using the package:

```R
# Install rextendr if not already installed
install.packages("rextendr")

# Install phylo2vec package
rextendr::document('./r-phylo2vec')

# Import the library
library('phylo2vec')

# A small demo
v = sample(5, FALSE)
to_newick(v)
```
