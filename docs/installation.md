# Installation

## Python package

### pip

The easiest way to install the standard Python package is using pip:

```bash
pip install phylo2vec
```

Several optimization schemes based on Phylo2Vec are also available, but require
extra dependencies. (See this
[notebook](https://phylo2vec.readthedocs.io/en/latest/demo_opt.html) for a
demo). To avoid bloating the standard package, these dependencies must be
installed separately. To do so, run:

```bash
pip install phylo2vec[opt]
```

### Manual installation

- We recommend setting up [pixi](https://pixi.sh/dev/) package management tool.
- Clone the repository and install using `pixi`:

```bash
git clone https://github.com/sbhattlab/phylo2vec.git
cd phylo2vec
pixi run -e py-phylo2vec install-python
```

This will compile and install the package as the core functionality is written
in Rust.

## R package

### Install a release (Windows, Mac, Ubuntu >= 22.04)

Retrieve one of the compiled binaries from the
[releases](https://github.com/sbhattlab/phylo2vec/releases) that fits your OS.
Once the file is downloaded, simply run `install.packages` in your R command
line.

```R
install.packages("/path/to/package_file", repos = NULL, type = 'source')
```

### `devtools`

⚠️ This requires installing [Rust](https://www.rust-lang.org/tools/install) to
build the core package.

```R
devtools::install_github("sbhattlab/phylo2vec", subdir="./r-phylo2vec", build = FALSE)
```

Note: to download a specific version, use:

```R
devtools::install_github("sbhattlab/phylo2vec@vX.Y.Z", subdir="./r-phylo2vec", build = FALSE)
```

### Manual installation

⚠️ This requires installing [Rust](https://www.rust-lang.org/tools/install) to
build the core package.

Clone the repository and run the following `install.packages` in your R command
line.

Note: to download a specific version, you can use `git checkout` to a desired
tag.

```bash
git clone https://github.com/sbhattlab/phylo2vec
cd phylo2vec
```

```R
install.packages("./r-phylo2vec", repos = NULL, type = 'source')
```
