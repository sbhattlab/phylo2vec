import sys
import os
import subprocess
import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.7+ via pip install tomli
    except ImportError:
        import toml as tomllib  # Fallback to toml library


def update_cargo_version(cargo_path: Path, new_version: str) -> None:
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
        if 'package' in toml_dict and 'version' in toml_dict['package']:
            old_version = toml_dict['package']['version']

            # If version hasn't changed, no need to update
            if old_version == new_version:
                print(f"Version is already set to {new_version}")
                return

            # Replace the version string in the content
            version_line = f'version = "{old_version}"'
            new_version_line = f'version = "{new_version}"'
            new_content = content.replace(version_line, new_version_line)

            # Write the updated content back to the file
            cargo_path.write_text(new_content)
            print(f"Successfully updated version from {old_version} to {new_version}")
        else:
            print("No version field found in the package section of the TOML file")

    except Exception as e:
        raise ValueError(f"Failed to parse or update TOML file: {e}")


def get_current_version(cargo_path: Path) -> str:
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
        if 'package' in toml_dict and 'version' in toml_dict['package']:
            return toml_dict['package']['version']
        else:
            raise ValueError("No version field found in the package section of the TOML file")
    except Exception as e:
        raise ValueError(f"Failed to parse TOML file: {e}")


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
    base_version = re.match(r'(\d+\.\d+\.\d+)', version_str).group(1)

    if not base_version:
        print(f"Warning: Could not parse version string '{version_str}'")
        return version_str

    # Split into major, minor, patch
    parts = base_version.split('.')
    if len(parts) != 3:
        print(f"Warning: Version '{base_version}' doesn't follow semver major.minor.patch format")
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
        # Clean version string for tag matching (remove any -rc. suffix)
        base_version = version_str
        if '-rc.' in base_version:
            base_version = base_version.split('-rc.')[0]

        # Format for tag search
        tag_prefix = f"v{base_version}"

        # First check if the tag exists
        result = subprocess.run(['git', 'tag', '-l', tag_prefix],  # skipcq: BAN-B607
                               capture_output=True, text=True, check=True)

        if not result.stdout.strip():
            print(f"Warning: No tag found matching '{tag_prefix}'")
            return "0"  # Return 0 if no tag found

        # Get the distance from tag to HEAD
        result = subprocess.run(['git', 'rev-list', f'{tag_prefix}..HEAD', '--count'],  # skipcq: BAN-B607
                               capture_output=True, text=True, check=True)
        distance = result.stdout.strip()
        return distance
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not calculate distance from version tag")
        return "0"  # Default to 0 if there's an error


def generate_version_with_distance(base_version: str) -> str:
    """
    Generate a new version string by incrementing the patch version and
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
    if '-rc.' in base_version:
        base_version = base_version.split('-rc.')[0]

    # Increment the patch version
    incremented_version = increment_patch_version(base_version)

    # Add the distance from version tag
    distance = get_distance_from_version_tag(base_version)
    return f"{incremented_version}-rc.{distance}" if distance else incremented_version


def extract_and_print_version(cargo_path: Path) -> str | None:
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
        version = get_current_version(cargo_path)
        print(f"Current version: {version}")

        # For GitHub Actions: set as output
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"version={version}\n")

        # For regular environments: set an environment variable
        os.environ['PHYLO2VEC_VERSION'] = version
        print(f"Set environment variable PHYLO2VEC_VERSION={version}")

        return version
    except Exception as e:
        print(f"Error extracting version: {e}")
        return None


def main():
    """
    Main function to update the version in Cargo.toml file.

    This function can either use a provided version from command line arguments,
    or generate a new version based on the current version plus distance from tag.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Exits with code 1 if there's an error.

    Usage:
    - With argument: python update_python_version.py NEW_VERSION
    - Without argument: python update_python_version.py
      (will generate version based on current version + tag distance)
    - With --extract-only: python update_python_version.py --extract-only
      (will only extract and print the version, without modifying the file)
    """
    cargo_path = Path(__file__).parents[1] / "py-phylo2vec" / "Cargo.toml"

    try:
        # Check if the user just wants to extract the version
        if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
            extract_and_print_version(cargo_path)
            return

        if len(sys.argv) > 1 and sys.argv[1] != "--extract-only":
            # Use provided version
            new_version = sys.argv[1]
        else:
            # Generate version with distance from version tag
            current_version = get_current_version(cargo_path)
            new_version = generate_version_with_distance(current_version)
            print(f"Generated new version: {new_version}")

        # Update the version in the Cargo.toml file
        update_cargo_version(cargo_path, new_version)

        # Extract and output the updated version
        extract_and_print_version(cargo_path)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
