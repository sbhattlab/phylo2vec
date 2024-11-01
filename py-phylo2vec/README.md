# py-phylo2vec

**NOTE: This is currently in active development and APIs will change. Use at your own risk.**

This directory contains the pylo2vec Python codebase, which includes Rust binding setup.

To install this python package, simply run the following within the root directory of
the repository with [`pixi`](https://pixi.sh/latest/):

```
pixi run -e py-phylo2vec install-python
```

Once installed you can try it by opening up `ipython` within the pixi
environment:

```
pixi run -e py-phylo2vec ipython
```

As of 10/18/2024 the phylo2vec Rust core code can be accesed from `_phylo2vec_core` module
within Python:

```
from phylo2vec import _phylo2vec_core
```

An example of calling `get_ancestry`:

```
_phylo2vec_core.get_ancestry([0,2,2,5,2])
```

