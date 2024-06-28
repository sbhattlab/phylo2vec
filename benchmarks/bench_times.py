"""
Benchmark execution times of p2v and ape (Paradis and Schliep, 2019) functions:
 * tree sampling (rtree)
 * pairwise cophenetic distance matrix (cophenetic.phylo)
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
from phylo2vec.metrics import cophenetic_distances
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
        "--sample-file",
        type=str,
        default="bench_sample_time",
        help="Output file name for sampling",
    )
    parser.add_argument(
        "--coph-file",
        type=str,
        default="bench_coph_time",
        help="Output file name for pairwise cophenetic distance",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Show plot output with matplotlib.",
    )

    return parser.parse_args()


def sample_p2v(all_leaves):
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


def sample_ape(all_leaves):
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


def coph_p2v(all_leaves):
    """Benchmark execution times of Phylo2Vec pairwise cophenetic distances

    Parameters
    ----------
    all_leaves : numpy.ndarray
        List of tree sizes (in number of leaves)

    Returns
    -------
    perfs
        Execution times in seconds for each tree size
    """
    print("Benchmark phylo2vec.metrics.cophenetic_distances...")

    perfs = np.zeros((len(all_leaves), N_REPEATS))

    for i, n_leaves in enumerate(all_leaves):
        # Compile
        v = sample(n_leaves)
        _ = cophenetic_distances(v)

        all_runs = np.array(
            timeit.repeat(
                f"cophenetic_distances(sample({n_leaves}))",
                "from phylo2vec.metrics import cophenetic_distances; from phylo2vec.utils import sample;"
                f"v = sample({n_leaves}); nw = cophenetic_distances(sample({n_leaves}));",
                number=N_TIMES,
                repeat=N_REPEATS,
            )
        )

        perfs[i, :] = all_runs

    return pd.Series(np.mean(perfs / N_TIMES, 1)).rename("Phylo2Vec")


def coph_ape(all_leaves):
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

        ro.globalenv["n_times"] = N_TIMES
        ro.globalenv["all_leaves"] = all_leaves

        perfs = pd.Series(
            ro.r(
                """
                perfs <- rep(0L, length(all_leaves))

                for (i in 1:length(all_leaves)) {
                    tr <- rtree(all_leaves[[i]], br = 1)
                    perfs[[i]] <- summary(
                        microbenchmark(
                            cophenetic.phylo(tr),
                            unit = "s",
                            times = 100L
                        )
                    )$mean
                }

                perfs
                """
            )
        ).rename("ape (cophenetic.phylo)")

    return perfs


def get_perfs(funcs, output_csv):
    all_leaves = np.arange(MIN_LEAVES, MAX_LEAVES, STEP_LEAVES)

    perfs = [func(all_leaves) for func in funcs]

    # Make DataFrame
    perfs_df = pd.concat(
        [
            *perfs,
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

    return perfs


def plot_perfs(perfs_df, output_pdf, show_plot=False):
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

    if show_plot:
        plt.show()


def _main_single(funcs, output_file, show_plot):
    output_csv = f"benchmarks/res/{output_file}.csv"
    output_pdf = f"benchmarks/img/{output_file}.pdf"

    perfs_df = get_perfs(funcs, output_csv)

    plot_perfs(perfs_df, output_pdf, show_plot=show_plot)


def main():
    args = parse_args()

    _main_single(
        funcs=(sample_ape, sample_p2v),
        output_file=args.sample_file,
        show_plot=args.show_plot,
    )

    _main_single(
        funcs=(coph_ape, coph_p2v), output_file=args.coph_file, show_plot=args.show_plot
    )


if __name__ == "__main__":
    main()
