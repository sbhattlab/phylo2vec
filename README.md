# Phylo2Vec

This repository contains an implementation of Phylo2Vec. It is distributed under the GNU Lesser General Public License v3.0 (LGPL).

[![PyPI version](https://badge.fury.io/py/phylo2vec.svg)](https://pypi.org/project/phylo2vec/)

Link to the preprint: <https://arxiv.org/abs/2304.12693>

## Installation

### Dependencies

* python>=3.9
* numba==0.56.4
* numpy==1.23.5
* biopython==1.80.0
* joblib==1.1.1
* ete3==3.1.3

### User installation

#### Pip

```bash
pip install phylo2vec
```

#### Manual installation

* We recommend to setup an isolated enviroment, using conda, mamba or virtualenv.
* Clone the repository and install using ```pip```:

```bash
git clone https://github.com/Neclow/phylo2vec_dev.git
pip install -e .
```

## Development

### Additional test dependencies

* ete3==3.1.3
* pytest==7.4.2
* six==1.16.0

### Testing

After installation, you can launch the test suite from outside the source directory (you will need to have pytest >= 7.4.2 installed):

```bash
pytest phylo2vec
```

## Basic usage

### Conversions

* The ```base``` module contains elements to convert a Newick string (```to_vector```) to a Phylo2Vec vector and vice versa (```to_newick```)

Example:

```python
import numpy as np
from phylo2vec.base import to_newick, to_vector

v = np.array([0, 1, 2, 3, 4])

newick = to_newick(v) # '(0,(1,(2,(3,(4,5)6)7)8)9)10;'

v_converted = to_vector(newick) # array([0, 1, 2, 3, 4], dtype=int16)
```

### Optimization

* The ```opt``` module contains methods to perform phylogenetic inference using Phylo2Vec vectors
* TODO: include GradME from <https://github.com/Neclow/GradME>

Example:

```python
from phylo2vec.opt import HillClimbingOptimizer

hc = HillClimbingOptimizer(raxml_cmd="/path/to/raxml-ng_v1.2.0_linux_x86_64/raxml-ng", verbose=True)
v_opt, taxa_dict, losses = hc.fit("/path/to/your_fasta_file.fa")
```

## Citation and other work

```latex
@article{phylo2vec,
  title={Phylo2Vec: a vector representation for binary trees},
  author={Penn, Matthew J and Scheidwasser, Neil and Khurana, Mark P and Duch{\^e}ne, David A and Donnelly, Christl A and Bhatt, Samir},
  journal={arXiv preprint arXiv:2304.12693},
  year={2023}
}
```

* Preprint repository (core functions are deprecated): <https://github.com/Neclow/phylo2vec_preprint>
* C++ version (deprecated): <https://github.com/Neclow/phylo2vec_cpp>
* GradME: <https://github.com/Neclow/GradME> = phylo2vec + minimum evolution + gradient descent
