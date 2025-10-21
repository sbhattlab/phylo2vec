# r-phylo2vec

**NOTE: This is currently in active development and APIs will change. Use at
your own risk.**

This directory contains the phylo2vec R codebase, which includes Rust binding
setup.

## Quick install and run

To quickly install the package and run it, simply run the following

```console
pixi run -e r-phylo2vec install-r
```

Once the package is installed you can open up the R terminal:

```console
pixi run -e r-phylo2vec R --interactive
```

In the R terminal, you can then import the `phylo2vec` library:

```R
library(phylo2vec)

# A small demo
v = sample_vector(10, FALSE)
to_newick(v)
```

## Development Guide

The core Rust code for phylo2vec is located in [phylo2vec](../phylo2vec)
directory. To add additional bindings for the Rust code, you can add more define
these in [`lib.rs`](./src/rust/src/lib.rs).

It uses [extendr](https://github.com/extendr/extendr), a safe and user-friendly
R extension interface using Rust. You should be able to directly import the
necessary functions from `phylo2vec` Rust crate.

**Note: You will need to run the steps below from the root of the repository for
it to work**

Open up `R` command line interactive mode within the pixi environment:

```console
pixi run -e r-phylo2vec R --interactive
```

### 1. Modify `src/rust/src/lib.rs`

Add the function with @export, so it will get exported from the generated R
package. (Without this tag, the function would be available internally for
package programming but not externally to users of the package.)

```Rust
/// Recover a rooted tree (in Newick format) from a Phylo2Vec v
/// @export
#[extendr]
fn to_newick(input_integers: Vec<i32>) -> String {
    let input_vector = input_integers.iter().map(|&x| x as usize).collect();
    let newick = ops::to_newick(&input_vector);
    newick
}
```

Don’t forget to add the function to `extendr_module!`:

```Rust
extendr_module! {
    mod phylo2vec;
    ...
    fn to_newick;
    ...
}
```

### 2. Run `rextendr::document` for building docs and binding R code

To rebuild the docs and binding code for R, you can simply run
`rextendr::document`:

```R
# Install rextendr if not already installed
install.packages("rextendr")

# Install phylo2vec package
rextendr::document('./r-phylo2vec')
```

### 3. Run `devtools::load_all` and test the function

Once in R, you can run the following to load and use the package:

```R
# Compile, install, and load phylo2vec package
devtools::load_all('./r-phylo2vec')

# A small demo
v = sample_vector(5, FALSE)
to_newick(v)
```
