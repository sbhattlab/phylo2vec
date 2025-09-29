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

Phylogenetics is a fundamental component of evolutionary analysis frameworks in biology [@yang2014] and linguistics [@atkinson2005]. Recently, the advent of large-scale genomics and the SARS-CoV-2 pandemic has highlighted the necessity for phylogenetic software to handle large datasets [@kapli2020; @attwood2022; @khurana2024; @kraemer2025]. While significant efforts have focused on scaling optimisation algorithms [@turakhia2021; @sanderson2021; @demaio2023], visualization [@sanderson2022], and lineage identification [@mcbroome2024], an emerging body of research has been dedicated to efficient representations of data for genomes [@deorowicz2023] and phylogenetic trees [@penn2024; @chauve2025; @richman2025]. Compared to the traditional Newick format which represents trees using strings of nested parentheses [@felsenstein2004], modern tree representations utilize integer vectors to define the tree topology traversal. This approach offers several advantages, including easier manipulation, increased memory efficiency, and applicability to machine learning.

Here, we present the latest release of \texttt{phylo2vec} (or Phylo2Vec), a high-performance software package for encoding, manipulating, and analysing binary phylogenetic trees. At its core, the package is based on the phylo2vec [@penn2024] representation of binary trees, and is designed to enable fast sampling and tree comparison. This release features a core implementation in Rust for improved performance and memory efficiency (\autoref{fig:benchmarks}), with wrappers in R and Python (superseding the original release [@penn2024]), making it accessible to a broad audience in the bioinformatics community.

![Benchmark times for converting a phylo2vec vector to a Newick string (left) and vice versa (right). Execution time was measured over at least 20 runs per size, comparing Python functions in the latest release (via Rust bindings with [PyO3](https://github.com/PyO3/pyo3)) against the previous release [@penn2024] based on \texttt{Numba} [@lam2015]. All benchmarks were performed on a workstation equipped with an AMD Ryzen Threadripper PRO 5995WX (64 cores, 2.7 GHz) and 256 GB of RAM. \label{fig:benchmarks}](fig1.pdf)

# Statement of need

The purpose of the \texttt{phylo2vec} library is threefold. First, it provides robust phylogenetic tree manipulation in Rust, complementing other efforts such as \texttt{light\_phylogeny} [@duchemin2018] for reconciled phylogenies [@nakhleh2013], and \texttt{rust-bio} [@koster2016], which does not yet cover phylogenetics. Second, it complements existing libraries such as \texttt{ape} [@paradis2019] in R, and \texttt{ete3} [@huerta2016] and \texttt{DendroPy} [@Moreno2024] in Python, by providing fast tree sampling, fast tree comparison and efficient tree data compression [@penn2024]. Third, the phylo2vec representation offers a pathway to using new optimisation frameworks for phylogenetic inference. A notable example is GradME [@penn2023], a gradient descent-based algorithm which uses a continuous relaxation of the phylo2vec representation.

# Features

The presented release of \texttt{phylo2vec} addresses optimisations limitations of [@penn2024] with $\mathcal{O}(n \log n)$ implementations for vector-to-Newick and Newick-to-vector conversions, leveraging Adelson-Velsky and Landis (AVL) trees [@adelson1962] and Fenwick trees [@fenwick1994], respectively.

New features include an extension of the vector representation to support branch length annotations, leaf-level operations (pruning, placement, MRCA identification), fast cophenetic distance matrix calculation, and various optimisation schemes based on phylo2vec tree representations, notably hill-climbing [@penn2024] and GradME [@penn2023]. We also propose a likelihood function for Bayesian MCMC inference that leverages tree representation similarities with \texttt{BEAGLE} [@suchard2009; @ayres2012]. Finally, user-friendliness is enhanced with step-by-step demos of phylo2vec’s representations and core functions.

# Maintenance

A strong focus of this release is to support long-term maintenance through implementing recommended software practices into its project structure and development workflow. The project is structured with a Rust API containing core algorithms with language bindings to avoid tight coupling and enable easy language additions. Additionally, we have established a robust continuous integration (CI) pipeline using GitHub Actions, which features:

* Unit test frameworks for Rust ([cargo](https://crates.io)), Python ([pytest](https://github.com/pytest-dev/pytest)), and R (testthat [@wickham2011])
* Benchmarking on the Rust code ([criterion](https://github.com/bheisler/criterion.rs)) and its Python bindings ([pytest-benchmark](https://github.com/ionelmc/pytest-benchmark))

Lastly, to complement Jupyter Notebook demos, comprehensive documentation is provided using [Jupyter Book](https://jupyterbook.org) and [Read The Docs](https://about.readthedocs.com/).

# Acknowledgements

S.B. acknowledges funding from the MRC Centre for Global Infectious Disease Analysis (reference MR/X020258/1), funded by the UK Medical Research Council (MRC). This UK funded award is carried out in the frame of the Global Health EDCTP3 Joint Undertaking. S.B. acknowledges support from the Danish National Research Foundation via a chair grant (DNRF160, also supporting N.S. and M.P.K.), The Eric and Wendy Schmidt Fund For Strategic Innovation via the Schmidt Polymath Award (G-22-63345, also supporting M.J.P. and F.M.A.), and the Novo Nordisk Foundation via The Novo Nordisk Young Investigator Award (NNF20OC0059309). D.A.D. is supported by a Data Science - Emerging researcher award from Novo Nordisk Fonden (NNF23OC0084647).

# References
