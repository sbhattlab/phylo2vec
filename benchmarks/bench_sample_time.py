"""
Comparison of tree sampling using the rtree
function from ape (Paradis and Schliep, 2019)
and Phylo2Vec in terms of execution time
"""

# pylint: disable=redefined-outer-name, protected-access
import timeit

from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns

from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from benchmarks.plot import clear_axes, set_size
from phylo2vec.base import to_newick
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

MIN_LEAVES, MAX_LEAVES, STEP_LEAVES = 5, 1000, 50

N_TIMES = 100

N_REPEATS = 7


def parse_args():
    """Parse optional arguments."""
    parser = ArgumentParser(description="Newick sampling time benchmark tool")
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Show plot output with matplotlib.",
    )
    parser.add_argument(
        "--output-file", type=str, default="bench_sample_time", help="Output file name"
    )

    return parser.parse_args()


def bench_p2v(all_leaves):
    """Benchmark execution times of Phylo2Vec tree sampling

    Parameters
    ----------
    all_leaves : numpy.ndarray
        List of tree sizes (in number of leaves)

    Returns
    -------
    perfs
        Execution times in seconds for each tree size
    """
    print("Benchmark phylo2vec.utils.sample...")

    perfs = np.zeros((len(all_leaves), N_REPEATS))

    for i, n_leaves in enumerate(all_leaves):
        # Compile
        _ = sample(n_leaves)
        _ = to_newick(sample(n_leaves))

        all_runs = np.array(
            timeit.repeat(
                f"to_newick(sample({n_leaves}))",
                "from phylo2vec.base import to_newick; from phylo2vec.utils import sample;"
                f"v = sample({n_leaves}); nw = to_newick(sample({n_leaves}));",
                number=N_TIMES,
                repeat=N_REPEATS,
            )
        )

        perfs[i, :] = all_runs

    return pd.Series(np.mean(perfs / N_TIMES, 1)).rename("Phylo2Vec")


def bench_ape(all_leaves):
    """Benchmark execution times of ape tree sampling

    Parameters
    ----------
    all_leaves : numpy.ndarray
        List of tree sizes (in number of leaves)

    Returns
    -------
    perfs
        Execution times in seconds for each tree size
    """
    print("Benchmark ape.rtree...")

    with localconverter(ro.default_converter + numpy2ri.converter):
        importr("ape")
        importr("microbenchmark")

        # ro.globalenv["n_leaves"] = n_leaves
        ro.globalenv["n_times"] = N_TIMES
        ro.globalenv["all_leaves"] = all_leaves

        perfs = pd.Series(
            ro.r(
                """
                perfs <- rep(0L, length(all_leaves))

                for (i in 1:length(all_leaves)) {
                    perfs[[i]] <- summary(
                        microbenchmark(
                            write.tree(rtree(all_leaves[[i]], br = NULL)),
                            unit = "s",
                            times = 100L
                        )
                    )$mean
                }

                perfs
                """
            )
        ).rename("ape (rtree)")

    return perfs


if __name__ == "__main__":
    args = parse_args()
    output_csv = f"benchmarks/res/{args.output_file}.csv"
    output_pdf = f"benchmarks/img/{args.output_file}.pdf"

    all_leaves = np.arange(MIN_LEAVES, MAX_LEAVES, STEP_LEAVES)

    # Make DataFrame
    perfs_df = pd.concat(
        [
            bench_p2v(all_leaves),
            bench_ape(all_leaves),
            pd.Series(all_leaves).rename("n_leaves"),
        ],
        axis=1,
    )

    perfs_df = perfs_df.melt(
        id_vars="n_leaves", var_name="generator", value_name="time [s]"
    )

    perfs_df["time [ms]"] = perfs_df["time [s]"].mul(1000)

    perfs_df.to_csv(output_csv, index=False)
    print(f"Data saved at {output_csv}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=set_size(290, "h"))

    sns.scatterplot(
        x="n_leaves",
        y="time [ms]",
        hue="generator",
        hue_order=["Phylo2Vec", "ape (rtree)"],
        data=perfs_df,
        palette={
            "Phylo2Vec": sns.color_palette("OrRd_r").as_hex()[0],
            "ape (rtree)": "k",
        },
        s=5,
        ax=ax,
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Time [ms]")
    ax.legend(title="")
    ax.set_title("Newick sampling")
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False

    clear_axes()

    plt.savefig(output_pdf, bbox_inches="tight", pad_inches=0)

    print(f"Plot saved at {output_pdf}")

    if args.show_plot:
        plt.show()
