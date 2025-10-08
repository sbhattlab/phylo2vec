# Phylo2Vec

[![PyPI version](https://badge.fury.io/py/phylo2vec.svg)](https://pypi.org/project/phylo2vec/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://phylo2vec.readthedocs.io)
[![DOI](https://zenodo.org/badge/710195598.svg)](https://zenodo.org/badge/latestdoi/710195598)

<span><img src="https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic"/><span>
![LGPL-3.0 License](https://badgen.net/badge/license/LGPL-3.0)

<!-- [![Documentation Status](https://readthedocs.org/projects/phylo2vec/badge/?version=latest)](https://phylo2vec.readthedocs.io/en/latest/?badge=latest) -->

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sbhattlab/phylo2vec/main.svg)](https://results.pre-commit.ci/latest/github/sbhattlab/phylo2vec/main)
[![CI Python](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-python.yml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-python.yml)
[![CI Rust](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-rust.yaml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-rust.yaml)
[![CI R](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-R.yml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-R.yml)

Phylo2Vec (or phylo2vec) is a high-performance software package for encoding,
manipulating, and analysing binary phylogenetic trees. At its core, the package
contains representation of binary trees, which defines a bijection from any tree
topology with 𝑛 leaves into an integer vector of size 𝑛 − 1. Compared to the
traditional Newick format, phylo2vec was designed with fast sampling, fast
conversion/compression from Newick-format trees to the Phylo2Vec format, and
rapid tree comparison in mind.

This current version features a core implementation in Rust, providing
significant performance improvements and memory efficiency while remaining
available in Python (superseding the version described in the
[original paper](https://doi.org/10.1093/sysbio/syae030)) and R via dedicated
wrappers, making it accessible to a broad audience in the bioinformatics
community.

Link to the paper:
[https://doi.org/10.1093/sysbio/syae030](https://doi.org/10.1093/sysbio/syae030)

## Installation

### Python package

#### Pip

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
pip install "phylo2vec[opt]"
```

#### Manual installation

- We recommend setting up [pixi](https://pixi.sh/dev/) package management tool.
- Clone the repository and install using `pixi`:

```bash
git clone https://github.com/sbhattlab/phylo2vec.git
cd phylo2vec
pixi run -e py-phylo2vec install-python
```

This will compile and install the package as the core functionality is written
in Rust.

### Installing R package

#### Option 1: from a release (Windows, Mac, Ubuntu >= 22.04)

Retrieve one of the compiled binaries from the
[releases](https://github.com/sbhattlab/phylo2vec/releases) that fits your OS.
Once the file is downloaded, simply run `install.packages` in your R command
line.

```R
install.packages("/path/to/package_file", repos = NULL, type = 'source')
```

#### Option 2: using `devtools`

⚠️ This requires installing [Rust](https://www.rust-lang.org/tools/install) to
build the core package.

```R
devtools::install_github("sbhattlab/phylo2vec", subdir="./r-phylo2vec", build = FALSE)
```

Note: to download a specific version, use:

```R
devtools::install_github("sbhattlab/phylo2vec@vX.Y.Z", subdir="./r-phylo2vec", build = FALSE)
```

#### Option 3: manual installation

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

## Basic Usage

### Python

#### Conversion between Newick and vector representations

```python
import numpy as np
from phylo2vec import from_newick, to_newick

# Convert a vector to Newick string
v = np.array([0, 1, 2, 3, 4])
newick = to_newick(v)  # '(0,(1,(2,(3,(4,5)6)7)8)9)10;'

# Convert Newick string back to vector
v_converted = from_newick(newick)  # array([0, 1, 2, 3, 4], dtype=int16)
```

#### Tree Manipulation

```python
from phylo2vec.utils.vector import add_leaf, remove_leaf, reroot_at_random

# Add a leaf to an existing tree
v_new = add_leaf(v, 2)  # Add a leaf to the third position

# Remove a leaf
v_reduced = remove_leaf(v, 1)  # Remove the second leaf

# Random rerooting
v_rerooted = reroot_at_random(v)
```

#### Optimization

To run the hill climbing-based optimisation scheme presented in the original
Phylo2Vec [paper](https://doi.org/10.1093/sysbio/syae030), run:

```python
# A hill-climbing scheme to optimize Phylo2Vec vectors
from phylo2vec.opt import HillClimbing

hc = HillClimbing(verbose=True)
hc_result = hc.fit("/path/to/your_fasta_file.fa")
```

#### Command-line interface (CLI)

We also provide a command-line interface for quick experimentation on
phylo2vec-derived objects.

To see the available functions, run:

```bash
phylo2vec --help
```

Examples:

```bash
phylo2vec samplev 5 # Sample a vector with 5 leaves
phylo2vec samplem 5 # Sample a matrix with 5 leaves
phylo2vec from_newick '((0,1),2);' # Convert a Newick to a vector
phylo2vec from_newick '((0:0.3,1:0.1):0.5,2:0.4);' # Convert a Newick to a matrix
phylo2vec to_newick 0,1,2 # Convert a vector to Newick
phylo2vec to_newick $'0.0,1.0,2.0\n0.0,3.0,4.0' # Convert a matrix to Newick
```

#### Datasets

Description of the datasets as well as download links are available in in the
[datasets](https://github.com/sbhattlab/phylo2vec/tree/main/py-phylo2vec/phylo2vec/datasets/descr)
directory.

Datasets for which a FASTA file is available can be downloaded and loaded into
Biopython:

```python
from phylo2vec.datasets import load_alignment

load_alignment("zika")
```

Readily downloadable datasets can be listed using:

```python
from phylo2vec.datasets import list_datasets

list_datasets()
```

## Documentation

For comprehensive documentation, tutorials, and API reference, visit:
[https://phylo2vec.readthedocs.io](https://phylo2vec.readthedocs.io)

## How to Contribute (issues, feature requests...)

Found a bug or want a new feature? We welcome contributions to phylo2vec! 🤗
Feel free to report any bugs or feature requests on our
[Issues page](https://github.com/sbhattlab/phylo2vec/issues). If you want to
contribute directly to the project, fork the repository, create a new branch,
and open a pull request (PR) on our
[Pull requests page](https://github.com/sbhattlab/phylo2vec/pulls).

Please refer to our
[Contributing guidelines](https://github.com/sbhattlab/phylo2vec/blob/main/CONTRIBUTING.md)
for more details how to report bugs, request features, or submit code
improvements.

Thanks to all our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=sbhattlab/phylo2vec)](https://github.com/sbhattlab/phylo2vec/graphs/contributors)

## License

This project is distributed under the
[GNU Lesser General Public License v3.0 (LGPL)](https://github.com/sbhattlab/phylo2vec/blob/main/LICENSE).

## Citation

If you use Phylo2Vec in your research, please cite:

```bibtex
@article{10.1093/sysbio/syae030,
    author = {Penn, Matthew J and Scheidwasser, Neil and Khurana, Mark P and Duchêne, David A and Donnelly, Christl A and Bhatt, Samir},
    title = {Phylo2Vec: a vector representation for binary trees},
    journal = {Systematic Biology},
    year = {2024},
    month = {03},
    doi = {10.1093/sysbio/syae030},
    url = {https://doi.org/10.1093/sysbio/syae030},
}
```

## Related Work

- Preprint repository (core functions are deprecated):
  [https://github.com/Neclow/phylo2vec_preprint](https://github.com/Neclow/phylo2vec_preprint)
- C++ version (deprecated):
  [https://github.com/Neclow/phylo2vec_cpp](https://github.com/Neclow/phylo2vec_cpp)
- GradME: [https://github.com/Neclow/GradME](https://github.com/Neclow/GradME) =
  phylo2vec + minimum evolution + gradient descent
