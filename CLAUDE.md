# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Phylo2Vec is a high-performance package for encoding, manipulating, and
analyzing binary phylogenetic trees. It defines a bijection from any tree
topology with n leaves into an integer vector of size n-1. The core is
implemented in Rust with bindings for Python (PyO3/maturin) and R (extendr).

## Common Commands

All commands use [pixi](https://pixi.sh/). The `-e` flag specifies the
environment.

### Building & Installing

```bash
pixi run build                          # Debug Rust build
pixi run build-release                  # Optimized release build
pixi run -e py-phylo2vec install-python # Install Python package (required before running Python tests)
pixi run -e r-phylo2vec install-r       # Install R package
```

### Testing

```bash
pixi run -e default test                # Rust tests (must specify environment)
pixi run -e py-phylo2vec test           # Python tests (all)
pixi run -e r-phylo2vec test            # R tests

# Run single Python test
pixi run -e py-phylo2vec pytest -k "test_name"
pixi run -e py-phylo2vec pytest ./py-phylo2vec/tests/test_base.py::test_function_name

# Run single Rust test with output
cargo test test_name -- --nocapture
```

### Linting & Formatting

```bash
pixi run fmt                            # cargo fmt
pixi run lint                           # cargo clippy (runs fmt first)
```

### Benchmarking & Profiling

```bash
pixi run benchmark                      # Rust benchmarks (Criterion)
pixi run -e py-phylo2vec benchmark      # Python benchmarks
pixi run profile [vector|matrix] [n_leaves]  # Profile with samply
```

## Architecture

### Codebase Structure

- `phylo2vec/` - Rust core library
  - `src/vector/` - Vector representation (base, convert, ops, graph, avl)
  - `src/matrix/` - Matrix representation (base, convert, ops, graph)
  - `src/newick/` - Newick format parsing
- `py-phylo2vec/` - Python package
  - `src/lib.rs` - PyO3 bindings exposed as `_phylo2vec_core`
  - `phylo2vec/` - Python API wrapping Rust bindings
- `r-phylo2vec/` - R package
  - `src/rust/` - extendr bindings
  - `R/` - R wrappers (structure mirrors Python)
    - `utils.R` - Tree operations (`sample_tree`, `get_common_ancestor`,
      `get_node_depth`, `get_node_depths`)
    - `stats.R` - Statistics (`cophenetic_distances`, `vcovp`, `precision`,
      `incidence`)
    - `base_newick.R` - Newick conversion (`to_newick`, `from_newick`)
    - `io.R` - File I/O
    - `extendr-wrappers.R` - Auto-generated Rust bindings

### Key Data Structures

**Vector** (`v`): Integer vector of length k = n_leaves - 1. Element `v[i]`
encodes which branch leaf `i+1` attaches to.

**Pairs** (`pairs`): `Vec<(usize, usize)>` of length k.
`pairs[i] = (child1, child2)` represents internal node `n_leaves + i` with its
two children.

**Ancestry** (`ancestry`): `Vec<[usize; 3]>` of length k.
`ancestry[i] = [child1, child2, parent]`. Ordered bottom-up (leaves to root).

**Branch lengths** (`bls`): `Vec<[f64; 2]>` of length k.
`bls[i] = [bl_to_child1, bl_to_child2]` for internal node `n_leaves + i`.

**Matrix** (`m`): 2D array of shape (k, 3). Column 0 is the vector, columns 1-2
are branch lengths.

### Key Patterns

**Rust Core Access**: Python modules import Rust functions via:

```python
from phylo2vec import _phylo2vec_core as core
```

**Adding Rust Functions**:

1. Implement in `phylo2vec/src/`
2. Add PyO3 binding in `py-phylo2vec/src/lib.rs` with `#[pyfunction]`
3. Create Python wrapper in `py-phylo2vec/phylo2vec/`
4. Add tests:
   - Python: `test_utils.py` for tree ops, `test_stats.py` for statistics
   - R: `test_utils.R` for tree ops (mirrors `utils.R`), `test_stats.R` for
     statistics (mirrors `stats.R`)

**Vector/Matrix API Pattern** (see `get_node_depth` in `vector/ops.rs` and
`matrix/ops.rs`):

- Implement generic internal function `_func(v, bls: Option<&Vec<[f64; 2]>>)` in
  `vector/` module
- When `bls` is `None`, use topological distances (all branch lengths = 1.0)
- Create thin wrapper in `matrix/` that parses matrix and calls generic function
  with `Some(&bls)`
- Python wrapper checks `ndim` and calls appropriate binding
- For purely topological functions (e.g., `get_common_ancestor`), the Python
  wrapper can extract column 0 from matrices

**R Wrapper Generation**: R wrappers in `r-phylo2vec/R/extendr-wrappers.R` are
auto-generated. After adding new Rust functions, run
`pixi run -e r-phylo2vec install-r` to regenerate.

**Cargo Workspace**: Three members - `phylo2vec`, `py-phylo2vec`,
`r-phylo2vec/src/rust`

## Performance Considerations

This library is built for efficiency. Avoid O(n²) or worse algorithms.

**Tree traversal patterns**:

- `get_ancestry_path_of_node`: O(n) traversal UP from node to root via pairs
- Avoid linear scans through ancestry for each node (causes O(n²))
- For repeated lookups, build HashMap<node, index> first

**When adding tree algorithms**:

- Prefer single-pass O(n) algorithms using pairs/ancestry ordering
- Ancestry is ordered bottom-up, enabling dynamic programming
- Test with n_leaves = 10, 100, 1000, 10000 to verify scaling

## Git Workflow

**Always create a new branch** for each feature or bugfix. Never commit directly
to `main`.

```bash
git checkout -b feat/feature-name    # For new features
git checkout -b fix/bug-description  # For bug fixes
```

## Commit Convention

Use [Conventional Commits](https://conventionalcommits.org) with Angular tags:

- `feat(rs):` / `feat(py):` / `feat(r):` - New features
- `fix:` - Bug fixes
- `docs:`, `test:`, `chore:`, `refactor:`
