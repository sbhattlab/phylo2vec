#!/usr/bin/env python3
"""
Phylo2Vec 0.1.0
https://github.com/Neclow/Phylo2Vec
Licensed under GNU Lesser General Public License v3.0
"""

from setuptools import find_packages, setup

setup(
    name="phylo2vec",
    version="0.1.0",
    description="Phylo2Vec: integer vector representation of binary (phylogenetic) trees",
    author="Neil Scheidwasser",
    author_email="neil.clow@sund.ku.dk",
    url="https://github.com/Neclow/phylo2vec",
    packages=find_packages(),
    python_requires="==3.10",
    install_requires=[
        "numba==0.56.4",
        "numpy==1.23.5",
        "biopython==1.80.0",
        "joblib==1.1.1",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ],
)
