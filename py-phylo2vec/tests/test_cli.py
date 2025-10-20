"""Tests for the phylo2vec CLI commands."""

from io import StringIO

from unittest.mock import patch

import numpy as np
import pytest

from phylo2vec.cli import parse_args, main


class TestMain:
    """Test CLI main."""

    def test_help(self):
        """Test main help message displays correctly."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "--help"]):
                parse_args()
        # Help should exit with code 0
        assert exc_info.value.code == 0

    def test_no_command(self):
        """Test that main with no command should redirect to help."""
        # Capture output from both commands
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.argv", ["phylo2vec"]):
                    parse_args()

        with patch("sys.stdout", new_callable=StringIO) as stdout_help:
            with pytest.raises(SystemExit) as exc_info_help:
                with patch("sys.argv", ["phylo2vec", "--help"]):
                    parse_args()

        # Both should exit with code 0
        assert exc_info.value.code == 0
        assert exc_info_help.value.code == 0

        assert stdout.getvalue() == stdout_help.getvalue()


class TestSampleV:
    """Test the samplev command."""

    def test_help(self):
        """Test samplev subcommand help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplev", "--help"]):
                parse_args()
        assert exc_info.value.code == 0

    def test_valid_number(self):
        """Test samplev with valid number of leaves."""
        with patch("sys.argv", ["phylo2vec", "samplev", "5"]):
            args = parse_args()
            assert hasattr(args, "n_leaves")
            assert args.n_leaves == 5
            assert hasattr(args, "ordered")
            assert not args.ordered

    def test_with_ordered_flag(self):
        """Test samplev with --ordered flag."""
        with patch("sys.argv", ["phylo2vec", "samplev", "3", "--ordered"]):
            args = parse_args()
            assert args.n_leaves == 3
            assert args.ordered

    def test_missing_argument(self):
        """Test samplev without required n_leaves argument."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplev"]):
                parse_args()
        assert exc_info.value.code != 0

    def test_string(self):
        """Test samplev with non-numeric string should fail."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stderr", new_callable=StringIO):  # Suppress error output
                with patch("sys.argv", ["phylo2vec", "samplev", "not_a_number"]):
                    parse_args()
        # Should exit with error code
        assert exc_info.value.code == 2

    def test_float(self):
        """Test samplev with float number should fail."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplev", "3.5"]):
                parse_args()
        # Should exit with error code
        assert exc_info.value.code == 2

    def test_negative_number(self):
        """Test samplev with negative number should fail."""
        with patch("sys.argv", ["phylo2vec", "samplev", "-5"]):
            with pytest.raises(ValueError):
                main()

    def test_zero(self):
        """Test samplev with negative number should fail."""
        with patch("sys.argv", ["phylo2vec", "samplev", "0"]):
            with pytest.raises(ValueError):
                main()

    def test_big_number(self, n_leaves=1000000):
        """Test samplev with a large number."""
        with patch("sys.argv", ["phylo2vec", "samplev", f"{1000000}"]):
            main()


class TestSampleM:
    """Test the samplem command."""

    def test_help(self):
        """Test samplem subcommand help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplem", "--help"]):
                parse_args()
        assert exc_info.value.code == 0

    def test_valid_number(self, n_leaves=5):
        """Test samplem with valid number of leaves."""
        with patch("sys.argv", ["phylo2vec", "samplem", f"{n_leaves}"]):
            args = parse_args()
            assert hasattr(args, "n_leaves")
            assert args.n_leaves == n_leaves
            assert hasattr(args, "ordered")
            assert not args.ordered

    def test_with_ordered_flag(self, n_leaves=3):
        """Test samplem with --ordered flag."""
        with patch("sys.argv", ["phylo2vec", "samplem", f"{n_leaves}", "--ordered"]):
            args = parse_args()
            assert args.n_leaves == n_leaves
            assert args.ordered

    def test_missing_argument(self):
        """Test samplem without required n_leaves argument."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplem"]):
                parse_args()
        assert exc_info.value.code != 0

    def test_string(self):
        """Test samplem with non-numeric string should fail."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stderr", new_callable=StringIO):  # Suppress error output
                with patch("sys.argv", ["phylo2vec", "samplem", "not_a_number"]):
                    parse_args()
        # Should exit with error code
        assert exc_info.value.code == 2

    def test_float(self):
        """Test samplem with float number should fail."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "samplem", "3.5"]):
                parse_args()
        # Should exit with error code
        assert exc_info.value.code == 2

    def test_negative_number(self):
        """Test samplem with negative number should fail."""
        with patch("sys.argv", ["phylo2vec", "samplem", "-5"]):
            with pytest.raises(ValueError):
                main()

    def test_zero(self):
        """Test samplem with zero should fail."""
        with patch("sys.argv", ["phylo2vec", "samplem", "0"]):
            with pytest.raises(ValueError):
                main()

    def test_big_number(self, n_leaves=100_000):
        """Test samplem with a large number."""
        with patch("sys.argv", ["phylo2vec", "samplem", f"{n_leaves}"]):
            main()


class TestFromNewick:
    """Test the from_newick command."""

    def test_help(self):
        """Test from_newick subcommand help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "from_newick", "--help"]):
                parse_args()
        assert exc_info.value.code == 0

    def test_valid_newick(self):
        """Test from_newick with valid newick string."""
        newick_str = "((0,1),2);"
        with patch("sys.argv", ["phylo2vec", "from_newick", newick_str]):
            args = parse_args()
            assert hasattr(args, "newick")
            assert args.newick == newick_str
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main()
                # Check that stdout == 0,2
                assert mock_stdout.getvalue().strip() == "0,2"

    def test_valid_newick2(self):
        """Test from_newick with valid newick string."""
        newick_str = "(((0,2)6,1)7,(3,4)5)8;"
        with patch("sys.argv", ["phylo2vec", "from_newick", newick_str]):
            args = parse_args()
            assert hasattr(args, "newick")
            assert args.newick == newick_str
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main()
                # Check that stdout == 0,2
                assert mock_stdout.getvalue().strip() == "0,0,4,3"

    def test_valid_newick_with_bls(self):
        """Test from_newick with valid newick string with BLs."""
        newick_str = "((0:0.3,1:0.1):0.5,2:0.4);"
        with patch("sys.argv", ["phylo2vec", "from_newick", newick_str]):
            args = parse_args()
            assert args.newick == newick_str
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main()
                arr = np.genfromtxt(
                    StringIO(mock_stdout.getvalue().strip()),
                    delimiter=",",
                    dtype=None,
                )
                assert np.array_equal(
                    arr,
                    np.array([[0.0, 0.3, 0.1], [2.0, 0.5, 0.4]]),
                )

    def test_empty_string(self):
        """Test from_newick with empty string."""
        with patch("sys.argv", ["phylo2vec", "from_newick", ""]):
            args = parse_args()
            assert args.newick == ""
            main()

    def test_missing_argument(self):
        """Test from_newick without required argument."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stderr", new_callable=StringIO):  # Suppress error output
                with patch("sys.argv", ["phylo2vec", "from_newick"]):
                    parse_args()
        assert exc_info.value.code == 2


class TestToNewick:
    """Test the to_newick command."""

    def test_help(self):
        """Test to_newick subcommand help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["phylo2vec", "to_newick", "--help"]):
                parse_args()
        assert exc_info.value.code == 0

    def test_valid_vector(self):
        """Test to_newick with valid vector input."""
        vector_input = "0,0,4,3"
        with patch("sys.argv", ["phylo2vec", "to_newick", vector_input]):
            args = parse_args()
            assert hasattr(args, "vector_or_matrix")
            assert args.vector_or_matrix == vector_input
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main()
                # Check that stdout == 0,2
                assert mock_stdout.getvalue().strip() == "(((0,2)6,1)7,(3,4)5)8;"

    def test_invalid_vector(self):
        """Test to_newick with invalid vector input."""
        with patch("sys.argv", ["phylo2vec", "to_newick", "1,2,3,4"]):
            with pytest.raises(AssertionError):
                main()

    def test_valid_matrix(self):
        """Test to_newick with valid matrix input."""
        matrix_input = "0.0,1.0,2.0\n0.0,3.0,4.0"  # real newline, not literal "\n"
        with patch("sys.argv", ["phylo2vec", "to_newick", matrix_input]):
            args = parse_args()
            assert hasattr(args, "vector_or_matrix")
            assert args.vector_or_matrix == matrix_input
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main()
                # Check that stdout == 0,2
                assert mock_stdout.getvalue().strip() == "((0:1.0,2:2.0)3:3.0,1:4.0)4;"

    def test_missing_argument(self):
        """Test to_newick without required argument."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stderr", new_callable=StringIO):  # Suppress error output
                with patch("sys.argv", ["phylo2vec", "to_newick"]):
                    parse_args()
        assert exc_info.value.code == 2
