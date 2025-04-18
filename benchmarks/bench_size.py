"""
Comparison of storage size of Phylo2Vec vectors vs. Newick strings
"""

# pylint: disable=protected-access
import sys

from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from benchmarks.plot import clear_axes, set_size
from phylo2vec.base import to_newick_from_vector
from phylo2vec.utils import sample_vector

MIN_LEAVES = 5
MAX_LEAVES = 10000
STEP_LEAVES = 50

tests = [
    "Phylo2Vec (int16)",
    "Phylo2Vec (int32)",
    "Phylo2Vec (string)",
    "Newick (string)",
]


def parse_args():
    """Parse optional arguments."""
    parser = ArgumentParser(description="Tree format size benchmark tool")
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Show plot output with matplotlib.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_false",
        help="Do not use LaTeX fonts for math in matplotlib",
    )
    parser.add_argument(
        "--output-file", type=str, default="bench_size", help="Output file name"
    )

    return parser.parse_args()


def main():
    """Main script"""
    args = parse_args()

    output_csv = f"benchmarks/res/{args.output_file}.csv"
    output_pdf = f"benchmarks/img/{args.output_file}.pdf"

    all_leaves = np.arange(MIN_LEAVES, MAX_LEAVES, STEP_LEAVES)

    # Pre-allocate size arrays for each test
    sizes = {test: np.zeros((len(all_leaves),), dtype=np.int32) for test in tests}

    # Compute sizes
    for i, n_leaves in tqdm(enumerate(all_leaves), total=len(all_leaves)):
        v = sample_vector(n_leaves)
        newick = to_newick_from_vector(v)

        sizes["Phylo2Vec (int16)"][i] = sys.getsizeof(v)
        sizes["Phylo2Vec (int32)"][i] = sys.getsizeof(v.astype(np.int32))
        sizes["Phylo2Vec (string)"][i] = sys.getsizeof(",".join(map(str, v)))
        sizes["Newick (string)"][i] = sys.getsizeof(newick)

    # Make a DataFrame and convert to long format
    sizes_df = pd.DataFrame(sizes)
    sizes_df["n_leaves"] = all_leaves

    sizes_df = sizes_df.melt(
        id_vars="n_leaves", var_name="format", value_name="Size [B]"
    )

    sizes_df["Size [kB]"] = sizes_df["Size [B]"].div(1000)

    sizes_df.to_csv(output_csv, index=False)
    print(f"Data saved at {output_csv}")

    if not args.no_latex:
        plt.rcParams.update(
            {
                "font.serif": ["sffamily"],
                "figure.dpi": "100",
                "font.size": 9,
                "text.usetex": True,
            }
        )

        matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")

    # Make the plot
    _, ax = plt.subplots(1, 1, figsize=set_size(290, "h"))

    palette = dict(zip(tests, [*sns.color_palette("OrRd_r", n_colors=3).as_hex(), "k"]))

    sns.scatterplot(
        x="n_leaves",
        y="Size [kB]",
        hue="format",
        data=sizes_df,
        s=4,
        palette=palette,
        ax=ax,
    )

    ax.legend(title="")
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False

    clear_axes()

    plt.savefig(output_pdf, bbox_inches="tight", pad_inches=0)

    print(f"Plot saved at {output_pdf}")

    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    main()
