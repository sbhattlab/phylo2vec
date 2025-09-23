# Contributing to Phylo2Vec

Thank you for your interest in contributing to Phylo2Vec! We welcome
contributions from the community to help improve our project.

## Getting started

To get started, please refer to our
[Development guide](https://phylo2vec.readthedocs.io/en/latest/development.html)
to familiarize yourself with our project’s structure, testing framework, and
benchmarking practices.

## Reporting Bugs and Requesting Features

We use GitHub Issues to track both bugs and feature requests.

If you encounter a bug or would like to suggest a new feature, please open a new
issue on the [Issues page](https://github.com/sbhattlab/phylo2vec/issues).

To help us handle issue swiftly, please include the following details:

- **Description**: A concise explanation of the problem or request, along with
  the expected outcome.
- **For bugs**: A minimal reproducible example and the complete error message or
  stack trace.
- **Environment details**: Your operating system and, if applicable, a list of
  relevant dependencies. Examples:
  - Python: NumPy, Biopython, pytest, ete
  - Rust: rand, ndarray, regex
  - R: ape, testthat

## Contributing to the codebase

### Working on the codebase

If you would like to contribute code directly, please follow these steps:

1. Fork the repository and create a branch from `main`:

```bash
git clone https://github.com/<your-username>/phylo2vec.git
cd phylo2vec
git checkout -b my-feature-branch
```

2. Make your changes, and add tests where applicable.

3. Run the test suite to ensure everything works as expected (see the
   [Testing guide](https://phylo2vec.readthedocs.io/en/latest/development.html#testing))

4. Push your changes to your new branch

```bash
git add file.xyz
git commit -m "Add new feature: XYZ"
git push origin my-feature-branch
```

### Submitting a pull request

To finalize your contribution to the codebase, open a pull request in the
[Pull requests page](https://github.com/Neclow/phylo2vec/pulls). Please adhere
to the following guidelines:

- Start your pull request title with a
  [conventional commit](https://conventionalcommits.org) tag following the
  [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).
- Use a descriptive title with active verbs to summarize your request
- In the main body, add any other information (link to an issue, snippets) that
  may help the maintainer process your request faster.

We appreciate your contributions and look forward to your involvement with this
project!
