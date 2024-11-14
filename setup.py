#!/usr/bin/env python3
"""
https://github.com/Neclow/Phylo2Vec
Licensed under GNU Lesser General Public License v3.0
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="phylo2vec",
    version="0.1.12",
    description="Phylo2Vec: integer vector representation of binary (phylogenetic) trees",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",  # This is important!
    author="Neil Scheidwasser",
    author_email="neil.clow@sund.ku.dk",
    url="https://github.com/Neclow/phylo2vec",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numba>=0.56.4",
        "numpy>=1.22,<2.1",
        "biopython==1.80.0",
        "joblib>=1.2.0",
        "ete3==3.1.3",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ],
)
