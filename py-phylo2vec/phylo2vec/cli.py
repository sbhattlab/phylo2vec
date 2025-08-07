"""
Command line interface for Phylo2Vec.

Examples:
phylo2vec samplev 5 # Sample a vector with 5 leaves
phylo2vec samplem 5 # Sample a matrix with 5 leaves
phylo2vec from_newick '((0,1),2);' # Convert a Newick to a vector
phylo2vec from_newick '((0:0.3,1:0.1):0.5,2:0.4);' # Convert a Newick to a matrix
phylo2vec to_newick 0,1,2 # Convert a vector to Newick
phylo2vec to_newick $'0.0,1.0,2.0\n0.0,3.0,4.0' # Convert a matrix to Newick
"""

import sys

from argparse import ArgumentParser

from phylo2vec.base.newick import from_newick, to_newick
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector
from phylo2vec.io.reader import read
from phylo2vec.io.writer import write


COMMANDS = {
    "samplev": {
        "help": "Sample a Phylo2Vec vector.",
        "func": sample_vector,
        "args": {
            "n_leaves": {
                "type": int,
                "help": "Number of leaves in the vector",
            },
            "--ordered": {
                "action": "store_true",
                "help": "Sample an ordered vector",
            },
        },
        "type": "write",
    },
    "samplem": {
        "help": "Sample a Phylo2Vec matrix.",
        "func": sample_matrix,
        "args": {
            "n_leaves": {
                "type": int,
                "help": "Number of leaves in the matrix",
            },
            "--ordered": {
                "action": "store_true",
                "help": "Sample an ordered matrix",
            },
        },
        "type": "write",
    },
    "from_newick": {
        "help": "Convert a Newick string to a Phylo2Vec vector/matrix.",
        "func": from_newick,
        "args": {
            "newick": {
                "type": str,
                "help": "Newick string representing the tree",
            },
        },
        "type": "write",
    },
    "to_newick": {
        "help": "Convert a Phylo2Vec vector/matrix to a Newick string.",
        "func": to_newick,
        "args": {
            "vector_or_matrix": {
                "type": str,
                "help": "Phylo2Vec vector/matrix to convert",
            },
        },
        "type": "read",
    },
}


def parse_args():
    """Phylo2vec argument parser.

    Sets up the command line interface for the Phylo2Vec CLI.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    global_parser = ArgumentParser(
        description="phylo2vec: a library for vector-based phylogenetic tree manipulation"
    )

    subparsers = global_parser.add_subparsers(title="subcommands")

    for command, details in COMMANDS.items():
        parser = subparsers.add_parser(command, help=details["help"])

        for arg_name, arg_details in details["args"].items():
            parser.add_argument(
                arg_name,
                **arg_details,
            )

    return global_parser.parse_args()


def main():
    """Main module"""

    args = parse_args()

    command_name = sys.argv[1] if len(sys.argv) > 1 else None

    if command_name is None:
        print("No command provided. Use --help for more information.")
        sys.exit(1)
    elif command_name not in COMMANDS:
        print(f"Unknown command: {command_name}. Use --help for more information.")
        sys.exit(1)
    else:
        command = COMMANDS[command_name]
        func = command["func"]
        the_args = vars(args)

        # Process input
        if command["type"] == "read":
            the_args["vector_or_matrix"] = read(the_args["vector_or_matrix"])

        out = func(**the_args)

        # Process output
        if command["type"] == "write":
            out = write(out)

        print(out)


if __name__ == "__main__":
    main()
