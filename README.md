# Phylo2Vec

[![PyPI version](https://badge.fury.io/py/phylo2vec.svg)](https://pypi.org/project/phylo2vec/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://phylo2vec.readthedocs.io)
[![DOI](https://zenodo.org/badge/710195598.svg)](https://zenodo.org/badge/latestdoi/710195598)

<span><img src="https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic" /><span>
![LGPL-3.0 License](https://badgen.net/badge/license/LGPL-3.0)

<!-- [![Documentation Status](https://readthedocs.org/projects/phylo2vec/badge/?version=latest)](https://phylo2vec.readthedocs.io/en/latest/?badge=latest) -->

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sbhattlab/phylo2vec/main.svg)](https://results.pre-commit.ci/latest/github/sbhattlab/phylo2vec/main)
[![CI Python](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-python.yml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-python.yml)
[![CI Rust](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-rust.yaml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-rust.yaml)
[![CI R](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-R.yml/badge.svg)](https://github.com/sbhattlab/phylo2vec/actions/workflows/ci-R.yml)

Phylo2Vec is a high-performance software package for encoding, manipulating, and
analyzing binary phylogenetic trees. At its core, the package contains
representation of binary trees, which defines a bijection from any tree topology
with 𝑛 leaves into an integer vector of size 𝑛 − 1. Compared to the traditional
Newick format, phylo2vec was designed with fast sampling and rapid tree
comparison in mind.

This current 2.0 version features a core implementation in Rust, providing
significant performance improvements and memory efficiency while remaining
available in Python (superseding the version described in the original paper
([Penn et al., 2024](https://doi.org/10.1093/sysbio/syae030))) and R via
dedicated wrappers, making it accessible to a broad audience in the
bioinformatics community.

Link to the paper:
[https://doi.org/10.1093/sysbio/syae030](https://doi.org/10.1093/sysbio/syae030)

## Installation

### Pip

The easiest way to install the Python package is using pip:

```bash
pip install phylo2vec
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

### Installing R package

To install the R package, first you need to retrieve one of the compiled file
within the package [releases](https://github.com/sbhattlab/phylo2vec/releases).
Once the file is downloaded, simply run `install.packages` in your R command
line.

```R
install.packages("/path/to/package_file", repos = NULL, type = 'source')
```

## Basic Usage

### Python

#### Conversion between Newick and vector representations

```python
import numpy as np
from phylo2vec.base import to_newick, to_vector

# Convert a vector to Newick string
v = np.array([0, 1, 2, 3, 4])
newick = to_newick(v)  # '(0,(1,(2,(3,(4,5)6)7)8)9)10;'

# Convert Newick string back to vector
v_converted = to_vector(newick)  # array([0, 1, 2, 3, 4], dtype=int16)
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

```python
from phylo2vec.opt import HillClimbingOptimizer

# Perform phylogenetic inference
hc = HillClimbingOptimizer(raxml_cmd="/path/to/raxml-ng", verbose=True)
v_opt, taxa_dict, losses = hc.fit("/path/to/your_fasta_file.fa")
```

## Documentation

For comprehensive documentation, tutorials, and API reference, visit:
[https://phylo2vec.readthedocs.io](https://phylo2vec.readthedocs.io)

## How to Contribute

We welcome contributions to Phylo2Vec! Here's how you can help:

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and add tests if applicable
3. **Run the tests** to ensure they pass
4. **Submit a pull request** with a detailed description of your changes

Please make sure to follow our coding standards and write appropriate tests for
new features.

Thanks to our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=sbhattlab/phylo2vec)](https://github.com/sbhattlab/phylo2vec/graphs/contributors)

## License

This project is distributed under the
[GNU Lesser General Public License v3.0 (LGPL)](./LICENSE).

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
