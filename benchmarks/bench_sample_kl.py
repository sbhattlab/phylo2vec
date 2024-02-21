"""
Comparison of tree sampling using the rmtree function from ape (Paradis and Schliep, 2019) and
Phylo2Vec in terms of divergence from perfect uniform sampling
"""

# pylint: disable=redefined-outer-name, protected-access
from argparse import ArgumentParser
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns

from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy import stats
from tqdm import tqdm

from benchmarks.plot import clear_axes, set_size
from phylo2vec.base import to_vector_no_parents
from phylo2vec.utils import sample

plt.rcParams.update(
    {
        "font.serif": ["sffamily"],
        "figure.dpi": "100",
        "font.size": 9,
        "text.usetex": True,
    }
)

matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")

N_TIMES = 1000

N_REPEATS = 50

MIN_LEAVES, MAX_LEAVES, STEP_LEAVES = 5, 55, 5


def parse_args():
    """Parse optional arguments."""
    parser = ArgumentParser(description="Newick sampling entropy benchmark tool")
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Show plot output with matplotlib.",
    )
    parser.add_argument(
        "--output-file", type=str, default="bench_sample_kl", help="Output file name"
    )

    return parser.parse_args()


@nb.njit
def to_integer(v):
    """Assign a integer to a Phylo2Vec vector.

    The integer-vector mapping is bijective and follows a
    similar method to that described by Rohlf (1983).

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec representation of a tree

    Returns
    -------
    res : int
        Corresponding integer
    """
    factor = 1.0
    res = 0.0
    for i in range(len(v)):
        res += v[len(v) - i - 1] * factor

        factor *= 2 * (len(v) - i) - 1
    return res


def generate_samples(n_repeats, n_times, min_leaves, max_leaves, step_leaves):
    """Generate tree samples (encoded as integers) using ape and Phylo2Vec

    Parameters
    ----------
    n_repeats : int
        Number of times the sampling loop is repeated
    n_times : int_
        Number of iterations in a sampling loop
    min_leaves : int
        Minimum number of leaves
    max_leaves : int
        Maximum number of leaves
    step_leaves : int
        Increment value in number of leaves

    Returns
    -------
    all_leaves : numpy.ndarray
        List of tree sizes (in number of leaves)
    all_ints_p2v : numpy.ndarray
        List of trees from Phylo2Vec converted into integer format
    all_ints_ape : numpy.ndarray
        List of trees from ape converted into integer format
    """
    all_leaves = np.arange(min_leaves, max_leaves, step_leaves)

    all_ints_p2v = np.zeros((len(all_leaves), n_repeats, n_times), dtype=np.float64)
    all_ints_ape = np.zeros((len(all_leaves), n_repeats, n_times), dtype=np.float64)

    for k, n_leaves in enumerate(all_leaves):
        print(f"n_leaves: {n_leaves}")
        for i in tqdm(range(n_repeats)):
            # Phylo2Vec integers
            for j in range(n_times):
                # Sample vectors with Phylo2Vec
                v = sample(n_leaves)

                all_ints_p2v[k, i, j] = to_integer(v)

            # Sample newicks with ape
            with localconverter(ro.default_converter + numpy2ri.converter):
                importr("ape")

                ro.globalenv["n_leaves"] = n_leaves
                ro.globalenv["n_times"] = n_times

                newicks = ro.r(
                    """
                    # tr = rmtree(n_times, n_leaves, br = NULL, equiprob = TRUE)
                    tr <- rmtopology(n_times, n_leaves, rooted = TRUE, equiprob = TRUE, br = NULL)
                    st <- rep("", n_times)
                    for(n in 1:length(tr)) {
                        tr[[n]]$tip.label <- gsub("t", "", tr[[n]]$tip.label)
                        tr[[n]]$tip.label <- as.character(as.numeric(tr[[n]]$tip.label)-1)
                        st[[n]] <- write.tree(tr[[n]])
                    }

                    st
                    """
                )

            # ape integers
            for j, nw in enumerate(newicks):
                # Convert to Phylo2Vec vector
                v = to_vector_no_parents(nw)

                all_ints_ape[k, i, j] = to_integer(v)

    return all_leaves, all_ints_p2v, all_ints_ape


def compute_entropies(all_leaves, all_ints_p2v, all_ints_ape):
    """Compute the KL-divergences of all_ints_p2v and all_ints_ape
    vs. a uniform distribution of ints for all tree sizes of interest

    Parameters
    ----------
    all_leaves : numpy.ndarray
        List of tree sizes (in number of leaves)
    all_ints_p2v : numpy.ndarray
        List of trees from Phylo2Vec converted into integer format
    all_ints_ape : numpy.ndarray
        List of trees from ape converted into integer format

    Returns
    -------
    entropies_df : pandas.DataFrame
        DataFrame with KL-divergences of Phylo2Vec and ape vs. unif distribution
        for all tree sizes
    """
    _, n_repeats, n_times = all_ints_p2v.shape

    entropies = {"Phylo2Vec": {}, "ape (rmtree)": {}}

    for k, n_leaves in enumerate(all_leaves):
        max_int = reduce(int.__mul__, range(2 * n_leaves - 3, 0, -2))

        entropies["Phylo2Vec"][n_leaves] = []
        entropies["ape (rmtree)"][n_leaves] = []

        # pylint: disable=protected-access
        for i in range(n_repeats):
            # Uniform PDF
            kde_unif = sns._statistics.KDE(clip=(0, max_int))(
                np.random.uniform(low=0, high=max_int, size=n_times)
            )[0]

            # P2V PDF
            kde_p2v = sns._statistics.KDE(clip=(0, max_int))(all_ints_p2v[k, i])[0]

            # APE PDF
            kde_ape = sns._statistics.KDE(clip=(0, max_int))(all_ints_ape[k, i])[0]

            # KL-divergence(P2V, Unif)
            entropies["Phylo2Vec"][n_leaves].append(
                stats.entropy(
                    kde_p2v,
                    kde_unif,
                )
            )

            # KL-divergence(ape, Unif)
            entropies["ape (rmtree)"][n_leaves].append(
                stats.entropy(
                    kde_ape,
                    kde_unif,
                )
            )
        # pylint: enable=protected-access

    entropies_df = (
        pd.DataFrame(entropies)
        .T.stack()
        .apply(pd.Series)
        .reset_index(names=["generator", "n_leaves"])
        .melt(id_vars=["generator", "n_leaves"], value_name="KL")
    )

    return entropies_df


if __name__ == "__main__":
    args = parse_args()

    output_csv = f"benchmarks/res/{args.output_file}.csv"
    output_pdf = f"benchmarks/img/{args.output_file}.pdf"

    print("Generating samples...")
    all_leaves, all_ints_p2v, all_ints_ape = generate_samples(
        n_times=N_TIMES,
        n_repeats=N_REPEATS,
        min_leaves=MIN_LEAVES,
        max_leaves=MAX_LEAVES,
        step_leaves=STEP_LEAVES,
    )

    print("Computing entropies (KL-divergence)")
    entropies = compute_entropies(
        all_leaves=all_leaves, all_ints_p2v=all_ints_p2v, all_ints_ape=all_ints_ape
    )

    entropies.to_csv(output_csv, index=False)
    print(f"Data saved at {output_csv}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=set_size(290, "h"))
    sns.lineplot(
        x="n_leaves",
        y="KL",
        hue="generator",
        data=entropies,
        palette={
            "Phylo2Vec": sns.color_palette("OrRd_r").as_hex()[0],
            "ape (rmtree)": "k",
        },
        ax=ax,
    )

    ax.set_ylabel("KL-divergence(generator, uniform)")
    ax.set_xlabel("Number of leaves")

    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.legend(title="")
    ax.set_title("Tree sampling")

    clear_axes()

    plt.savefig(output_pdf, bbox_inches="tight", pad_inches=0)

    print(f"Plot saved at {output_pdf}")

    if args.show_plot:
        plt.show()
