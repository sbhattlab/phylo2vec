---
title: 'phylo2vec: a library for vector-based phylogenetic tree manipulation'
tags:
  - Rust
  - Python
  - R
  - bioinformatics
  - phylogenetics
  - binary tree
authors:
  - name: Neil Scheidwasser
    orcid: 0000-0001-9922-0289
    affiliation: 1 # (Multiple affiliations must be quoted)
    equal-contrib: true
  - name: Ayush Nag
    affiliation: 2
    equal-contrib: true
  - name: Matthew J Penn
    orcid: 0000-0001-8682-5393
    affiliation: 1
  - name: Anthony MV Jakob
    orcid: 0000-0002-0996-1356
    affiliation: 4
  - name: Frederik Mølkjær Andersen
    orcid: 0009-0004-4071-3707
    affiliation: 1
  - name: Mark P Khurana
    orcid: 0000-0002-1123-7674
    affiliation: 1
  - name: Landung Setiawan
    orcid: 0000-0002-1624-2667
    affiliation: 2
  - name: Madeline Gordon
    orcid: 0009-0003-6220-7218
    affiliation: 2
  - name: David A Duchêne
    orcid: 0000-0002-5479-1974
    affiliation: 1
  - name: Samir Bhatt
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 3"
    orcid: 0000-0002-0891-4611
affiliations:
 - name: Section of Health Data Science and AI, University of Copenhagen, Copenhagen, Denmark
   index: 1
 - name: eScience Institute, University of Washington, Seattle, United States
   index: 2
 - name: MRC Centre for Global Infectious Disease Analysis, Imperial College London, London, United Kingdom
   index: 3
 - name: Independent researcher
   index: 4
date: 14 May 2025
geometry: margin=2cm
bibliography: paper.bib
---

# Summary

Phylogenetics is a fundamental component of many analysis frameworks in biology [@yang2014] as well as linguistics [@atkinson2005] to study the evolutionary relationships of different entities. Recently, the advent of large-scale genomics and the SARS-CoV-2 pandemic has underscored the necessity for phylogenetic software to handle large datasets of genomes or phylogenetic trees [@kapli2020; @attwood2022; @khurana2024; @kraemer2025]. While significant efforts have focused on scaling optimisation algorithms [@turakhia2021; @sanderson2021; @demaio2023], visualization [@sanderson2022], and lineage identification [@mcbroome2024], an emerging body of research has been dedicated to efficient representations of data for genomes [@deorowicz2023] and phylogenetic trees such as phylo2vec [@penn2024; @chauve2025; @richman2025]. Compared to traditional tree representations such as the Newick format [@felsenstein2004], which represents trees using strings of nested parentheses, modern representations of phylogenetic trees utilize integer vectors to define the tree topology traversal. This approach offers several advantages, including easier manipulability, increased memory efficiency, and applicability to downstream tasks such as machine learning.

Here, we present the latest release of \texttt{phylo2vec} (or Phylo2Vec), a high-performance software package for encoding, manipulating, and analysing binary phylogenetic trees. At its core, the package is based on the phylo2vec [@penn2024] representation of binary trees, which defines a bijection from any tree topology with $n$ leaves into an integer vector of size $n-1$. Compared to the traditional Newick format, phylo2vec is designed to enable fast sampling and comparison of binary trees. This release features a core implementation in Rust, providing significant performance improvements and memory efficiency (\autoref{fig:benchmarks}), while remaining available in Python (superseding the release described in the original paper [@penn2024]) and R via dedicated wrappers, making it accessible to a broad audience in the bioinformatics community.

![Benchmark times for converting a phylo2vec vector to a Newick string (left) and vice versa (right). For each size, we evaluated the execution time for a minimum of 20 rounds using \texttt{pytest-benchmark}. We compare the execution time of the Python functions in the latest release, which rely on Rust bindings via [PyO3](https://github.com/PyO3/pyo3), with the previous release [@penn2024], which make use of just-in-time (JIT) compilation of Python functions using \texttt{Numba} [@lam2015]. Execution times were benchmarked
on an AMD Ryzen Threadripper PRO 5995WX with a 64-core CPU @ 7 GHz and 256 GB of RAM. \label{fig:benchmarks}](fig1.pdf)

# Statement of need

The purpose of the \texttt{phylo2vec} library is threefold. First, the core of the library aims at providing a robust phylogenetic tree manipulation library in Rust, complementing other efforts such as \texttt{light\_phylogeny} [@duchemin2018], which focuses on tree visualization and manipulation of reconciled phylogenies [@nakhleh2013], and \texttt{rust-bio} [@koster2016], a comprehensive bioinformatics library which does not yet cover phylogenetics. Second, \texttt{phylo2vec} aims at complementing existing phylogenetic libraries such as \texttt{ape} [@paradis2019] in R, and \texttt{ete3} [@huerta2016] and \texttt{DendroPy} [@Moreno2024] in Python, by providing fast tree sampling, fast tree comparison and efficient tree data compression [@penn2024]. Third, the inherent tree representation of phylo2vec offers a pathway to several optimisation frameworks for phylogenetic inference. A notable example is GradME [@penn2023], a gradient descent-based algorithm which uses a relaxed representation of phylo2vec into a continuous space.

# Features

The presented release of \texttt{phylo2vec} addresses several limitations of [@penn2024]. From an optimisation perspective, it introduces efficient $\mathcal{O}(n \log n)$ implementations for both vector-to-Newick and Newick-to-vector conversions, leveraging Adelson-Velsky and Landis (AVL) trees [@adelson1962] and Fenwick trees [@fenwick1994], respectively.

In terms of new features, it supports branch length annotations by extending the vector representation of size $n-1$ to a matrix of size $(n-1) \times 3$, where $n$ denotes the number of leaves (or taxa) in a tree (\autoref{fig:phylo2mat}). Further enhancements include several leaf-level operations (pruning, placement, MRCA identification), fast cophenetic distance matrix calculation, and various optimisation schemes based on phylo2vec tree representations, notably hill-climbing and GradME [@penn2023]. Bayesian phylogenetic inference via Markov Chain Monte Carlo (MCMC) with phylo2vec matrices is also made possible by a likelihood function that leverages tree representation similarities between phylo2vec and \texttt{BEAGLE}, a highly optimised library that underpins a number of phylogenetic software [@suchard2009; @ayres2012]. Finally, user-friendliness is enhanced by step-by-step demos of the inner workings of phylo2vec's representations and their conversion to commonly used phylogenetic tree formats (e.g., Newick string, list of edges).

![Recovering a tree from a phylo2vec vector: example for $\boldsymbol{v} = [0, 2, 0, 4]$. (a) Main algorithm for leaf placement described in [@penn2024]. (b) Augmenting the phylo2vec vector into a matrix $\boldsymbol{m}$ with branch lengths. We use an intermediary ancestry matrix whereby each row describes a cherry (two children nodes and the parent node), which we augment with two columns of branch lengths. The node with the smallest descendant will have the branch length in the 2nd column, and the other the branch length in the 3rd column.\label{fig:phylo2mat}](fig2.pdf)

# Maintenance

With Phylo2Vec, we aim to support long-term maintenance through implementing recommended software practices explicitly into the structure of the project and development workflow. More specifically, we have structured the project such that the Rust API contains the core algorithms, and all other language components are APIs that bind to the Rust functions. This avoids tight coupling, as it allows for the possibility of adding new languages to bind to the Rust API's, without needing to change anything in the Rust project itself. Additionally, we have established a robust continuous integration (CI) pipeline using GitHub Actions, which features:

* Unit test frameworks for Rust ([cargo](https://crates.io)), Python ([pytest](https://github.com/pytest-dev/pytest)), and R (testthat [@wickham2011])
* Benchmarking on the Rust code ([criterion](https://github.com/bheisler/criterion.rs)) and its Python bindings ([pytest-benchmark](https://github.com/ionelmc/pytest-benchmark))

Lastly, to complement Jupyter Notebook demos, comprehensive documentation is provided using [Jupyter Book](https://jupyterbook.org) and [Read The Docs](https://about.readthedocs.com/).

# Acknowledgements

S.B. acknowledges funding from the MRC Centre for Global Infectious Disease Analysis (reference MR/X020258/1), funded by the UK Medical Research Council (MRC). This UK funded award is carried out in the frame of the Global Health EDCTP3 Joint Undertaking. S.B. acknowledges support from the Danish National Research Foundation via a chair grant (DNRF160) which also supports N.S. and M.P.K. S.B. acknowledges support from The Eric and Wendy Schmidt Fund For Strategic Innovation via the Schmidt Polymath Award (G-22-63345) which also supports M.J.P. and F.M.A. S.B. acknowledges support from the Novo Nordisk Foundation via The Novo Nordisk Young Investigator Award (NNF20OC0059309). D.A.D. is supported by a Data Science - Emerging researcher award from Novo Nordisk Fonden (NNF23OC0084647).

# References
