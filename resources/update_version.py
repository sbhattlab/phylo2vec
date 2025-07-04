import argparse
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.7+ via pip install tomli
    except ImportError:
        import toml as tomllib  # Fallback to toml library


# ---- Common utility functions ----


def increment_patch_version(version_str: str) -> str:
    """
    Increment the patch version of a semver string.

    Parameters
    ----------
    version_str : str
        Version string in format 'major.minor.patch' or with additional suffixes

    Returns
    -------
    str
        Version with incremented patch number
    """
    # Extract the base version (remove any suffixes like -rc.xxx or .devxxx)
    match = re.match(r"(\d+\.\d+\.\d+)", version_str)

    if not match:
        print(f"Warning: Could not parse version string '{version_str}'")
        return version_str

    base_version = match.group(1)

    # Split into major, minor, patch
    parts = base_version.split(".")
    if len(parts) != 3:
        print(
            f"Warning: Version '{base_version}' doesn't follow semver major.minor.patch format"
        )
        return version_str

    # Increment patch version
    try:
        major, minor, patch = parts
        new_patch = int(patch) + 1
        return f"{major}.{minor}.{new_patch}"
    except (ValueError, IndexError) as e:
        print(f"Warning: Failed to increment patch version: {e}")
        return version_str


def get_distance_from_version_tag(version_str):
    """
    Get the distance (number of commits) from the tag matching the current version.

    Parameters
    ----------
    version_str : str
        The current version string to find matching tag

    Returns
    -------
    str
        The distance from the tag, or an empty string if git command fails
    """
    try:
        git_executable = shutil.which("git")
        if not git_executable:
            raise RuntimeError("`git` command not found. Please install Git.")

        # Clean version string for tag matching (remove any suffixes)
        match = re.match(r"(\d+\.\d+\.\d+)", version_str)

        if not match:
            print(f"Warning: Could not parse version string '{version_str}'")
            return version_str

        base_version = match.group(1)

        # Format for tag search
        tag_prefix = f"v{base_version}"

        # First check if the tag exists
        result = subprocess.run(
            [git_executable, "tag", "-l", tag_prefix],  # skipcq: BAN-B607
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print(f"Warning: No tag found matching '{tag_prefix}'")
            return "0"  # Return 0 if no tag found

        # Get the distance from tag to HEAD
        result = subprocess.run(
            [git_executable, "rev-list", f"{tag_prefix}..HEAD", "--count"],
            capture_output=True,
            text=True,
            check=True,
        )
        distance = result.stdout.strip()
        return distance
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not calculate distance from version tag")
        return "0"  # Default to 0 if there's an error


# ---- Python/Cargo.toml handling ----


def get_python_cargo_path() -> Path:
    """Get the path to the Python binding's Cargo.toml file."""
    return Path(__file__).parents[1] / "py-phylo2vec" / "Cargo.toml"


def get_python_version(cargo_path: Path) -> str:
    """
    Get the current version from the Cargo.toml file.

    Parameters
    ----------
    cargo_path : Path
        Path to the Cargo.toml file

    Returns
    -------
    str
        Current version string

    Raises
    ------
    FileNotFoundError
        If the Cargo.toml file does not exist at the specified path
    ValueError
        If the TOML file cannot be parsed or the version is not found
    """
    if not cargo_path.exists():
        raise FileNotFoundError(f"Cargo.toml not found at {cargo_path}")

    content = cargo_path.read_text()

    try:
        toml_dict = tomllib.loads(content)
        if "package" in toml_dict and "version" in toml_dict["package"]:
            return toml_dict["package"]["version"]
        raise ValueError(
            "No version field found in the package section of the TOML file"
        )
    except Exception as err:
        raise ValueError(f"Failed to parse TOML file: {err}") from err


def update_python_version(cargo_path: Path, new_version: str) -> None:
    """
    Update the version string in Cargo.toml file using TOML library.

    Parameters
    ----------
    cargo_path : Path
        Path to the Cargo.toml file
    new_version : str
        New version string to set

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the Cargo.toml file does not exist at the specified path
    ValueError
        If the TOML file cannot be parsed or updated
    """
    if not cargo_path.exists():
        raise FileNotFoundError(f"Cargo.toml not found at {cargo_path}")

    # Read the TOML file as text first
    content = cargo_path.read_text()

    # Parse the TOML content
    try:
        # tomllib is read-only, so we need to modify the content as string
        # and write it back manually
        toml_dict = tomllib.loads(content)

        # Check if the version key exists in the package section
        if "package" in toml_dict and "version" in toml_dict["package"]:
            old_version = toml_dict["package"]["version"]

            # If version hasn't changed, no need to update
            if old_version == new_version:
                print(f"Python version is already set to {new_version}")
                return

            # Replace the version string in the content
            version_line = f'version = "{old_version}"'
            new_version_line = f'version = "{new_version}"'
            new_content = content.replace(version_line, new_version_line)

            # Write the updated content back to the file
            cargo_path.write_text(new_content)
            print(
                f"Successfully updated Python version from {old_version} to {new_version}"
            )
        else:
            print("No version field found in the package section of the TOML file")

    except Exception as err:
        raise ValueError(f"Failed to parse or update TOML file: {err}") from err


def generate_python_version_with_distance(base_version: str) -> str:
    """
    Generate a new version string for Python by incrementing the patch version and
    appending the distance from the current version tag.

    Parameters
    ----------
    base_version : str
        Base version string (e.g., '0.1.12')

    Returns
    -------
    str
        New version string with incremented patch and tag distance (e.g., '0.1.13-rc.5')
        where 5 is the number of commits since the tag matching base_version
    """
    # Clean the base version first (remove any existing rc or dev tags)
    if "-rc." in base_version:
        base_version = base_version.split("-rc.")[0]

    # Increment the patch version
    incremented_version = increment_patch_version(base_version)

    # Add the distance from version tag
    distance = get_distance_from_version_tag(base_version)
    return f"{incremented_version}-rc.{distance}" if distance else incremented_version


def extract_and_print_python_version(cargo_path: Path) -> str | None:
    """
    Extract the version from the Cargo.toml file and print it to stdout.
    This is cross-platform compatible (works on Windows, macOS, Linux).

    Parameters
    ----------
    cargo_path : Path
        Path to the Cargo.toml file

    Returns
    -------
    str
        The current version string
    """
    try:
        version = get_python_version(cargo_path)
        print(f"Current Python version: {version}")

        # For GitHub Actions: set as output
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"version={version}\n")

        # For regular environments: set an environment variable
        os.environ["PHYLO2VEC_VERSION"] = version
        print(f"Set environment variable PHYLO2VEC_VERSION={version}")

        return version
    except Exception as e:
        print(f"Error extracting Python version: {e}")
        return None


# ---- R/DESCRIPTION handling ----


def get_r_description_path() -> Path:
    """Get the path to the R package's DESCRIPTION file."""
    return Path(__file__).parents[1] / "r-phylo2vec" / "DESCRIPTION"


def get_r_version(description_path: Path) -> str:
    """
    Get the current version from the DESCRIPTION file.

    Parameters
    ----------
    description_path : Path
        Path to the DESCRIPTION file

    Returns
    -------
    str
        Current version string

    Raises
    ------
    FileNotFoundError
        If the DESCRIPTION file does not exist at the specified path
    ValueError
        If the DESCRIPTION file cannot be parsed or the version is not found
    """
    if not description_path.exists():
        raise FileNotFoundError(f"DESCRIPTION file not found at {description_path}")

    content = description_path.read_text()

    # Find the current version line
    version_pattern = re.compile(r"^Version:\s*([\d\.]+)", re.MULTILINE)
    match = version_pattern.search(content)

    if not match:
        raise ValueError("No Version field found in the DESCRIPTION file")

    return match.group(1)


def update_r_version(description_path: Path, new_version: str) -> None:
    """
    Update the version string in R DESCRIPTION file.

    Parameters
    ----------
    description_path : Path
        Path to the DESCRIPTION file
    new_version : str
        New version string to set

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the DESCRIPTION file does not exist at the specified path
    ValueError
        If the DESCRIPTION file cannot be parsed or updated
    """
    if not description_path.exists():
        raise FileNotFoundError(f"DESCRIPTION file not found at {description_path}")

    # Read the DESCRIPTION file
    content = description_path.read_text()

    # Find the current version line
    version_pattern = re.compile(r"^Version:\s*([\d\.]+)", re.MULTILINE)
    match = version_pattern.search(content)

    if not match:
        raise ValueError("No Version field found in the DESCRIPTION file")

    old_version = match.group(1)

    # If version hasn't changed, no need to update
    if old_version == new_version:
        print(f"R version is already set to {new_version}")
        return

    # Replace the version string in the content
    new_content = version_pattern.sub(f"Version: {new_version}", content)

    # Write the updated content back to the file
    description_path.write_text(new_content)
    print(f"Successfully updated R version from {old_version} to {new_version}")


def generate_r_version_with_distance(base_version: str) -> str:
    """
    Generate a new version string for R by incrementing the patch version and
    appending the distance from the current version tag as a fourth digit.

    Parameters
    ----------
    base_version : str
        Base version string (e.g., '0.1.12' or '0.1.12.5')

    Returns
    -------
    str
        New version string with incremented patch and tag distance as fourth digit
        (e.g., '0.1.13.5') where 5 is the number of commits since the tag matching
        the original base version
    """
    # Clean the base version first (remove any existing fourth digit)
    base_parts = base_version.split(".")
    if len(base_parts) > 3:
        base_version = ".".join(base_parts[:3])

    # Increment the patch version
    incremented_version = increment_patch_version(base_version)

    # Add the distance from version tag as the fourth digit
    distance = get_distance_from_version_tag(base_version)
    return (
        f"{incremented_version}.{distance}"
        if distance and distance != "0"
        else incremented_version
    )


def extract_and_print_r_version(description_path: Path) -> str | None:
    """
    Extract the version from the DESCRIPTION file and print it to stdout.
    This is cross-platform compatible (works on Windows, macOS, Linux).

    Parameters
    ----------
    description_path : Path
        Path to the DESCRIPTION file

    Returns
    -------
    str
        The current version string
    """
    try:
        version = get_r_version(description_path)
        print(f"Current R version: {version}")

        # For GitHub Actions: set as output
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"r_version={version}\n")

        # For regular environments: set an environment variable
        os.environ["PHYLO2VEC_R_VERSION"] = version
        print(f"Set environment variable PHYLO2VEC_R_VERSION={version}")

        return version
    except Exception as e:
        print(f"Error extracting R version: {e}")
        return None


# ---- Main functions ----


def process_python_version(
    new_version: str = None, extract_only: bool = False
) -> str | None:
    """Process the Python package version."""
    cargo_path = get_python_cargo_path()

    try:
        if extract_only:
            return extract_and_print_python_version(cargo_path)

        if new_version:
            # Use provided version
            pass
        else:
            # Generate version with distance from version tag
            current_version = get_python_version(cargo_path)
            new_version = generate_python_version_with_distance(current_version)
            print(f"Generated new Python version: {new_version}")

        # Update the version in the Cargo.toml file
        update_python_version(cargo_path, new_version)

        # Extract and output the updated version
        return extract_and_print_python_version(cargo_path)

    except Exception as e:
        print(f"Error processing Python version: {e}")
        return None


def process_r_version(
    new_version: str = None, extract_only: bool = False
) -> str | None:
    """Process the R package version."""
    description_path = get_r_description_path()

    try:
        if extract_only:
            return extract_and_print_r_version(description_path)

        if new_version:
            # Use provided version
            pass
        else:
            # Generate version with distance from version tag
            current_version = get_r_version(description_path)
            new_version = generate_r_version_with_distance(current_version)
            print(f"Generated new R version: {new_version}")

        # Update the version in the DESCRIPTION file
        update_r_version(description_path, new_version)

        # Extract and output the updated version
        return extract_and_print_r_version(description_path)

    except Exception as e:
        print(f"Error processing R version: {e}")
        return None


def parse_args() -> Tuple[bool, bool, str, bool]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update version numbers for phylo2vec packages.",
        epilog="""
        Examples:
        - python update_version.py --python --r
          (update both Python and R versions)
        - python update_version.py --python 0.2.1 --r 0.2.1
          (set specific version for both)
        - python update_version.py --python --extract-only
          (extract Python version without modification)
        """,
    )

    parser.add_argument(
        "--python", action="store_true", help="Update Python package version"
    )
    parser.add_argument("--r", action="store_true", help="Update R package version")
    parser.add_argument(
        "--version",
        type=str,
        help="Set a specific version (applied to all selected packages)",
        required=False,
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract and print versions, no modifications",
    )

    args = parser.parse_args()

    # Default to both if none specified
    if not args.python and not args.r:
        args.python = True
        args.r = True

    return args.python, args.r, args.version, args.extract_only


def main():
    """
    Main function to update the version in Python and/or R packages.

    This function handles both Python and R package versioning based on command line arguments.
    """
    try:
        update_python, update_r, new_version, extract_only = parse_args()

        if update_python:
            process_python_version(new_version, extract_only)

        if update_r:
            process_r_version(new_version, extract_only)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
