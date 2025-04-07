
import sys
import datetime
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


def generate_version_with_timestamp(base_version: str) -> str:
    """
    Generate a new version string by appending the current date and time to the base version.

    Parameters
    ----------
    base_version : str
        Base version string (e.g., '0.1.12')

    Returns
    -------
    str
        New version string with timestamp (e.g., '0.1.12.dev202504071437')
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")  # Format: YYYYMMDDHHMM (up to minute)

    # Remove any existing dev version if present
    if '.dev' in base_version:
        base_version = base_version.split('.dev')[0]

    return f"{base_version}.dev{timestamp}"


def main():
    """
    Main function to update the version in Cargo.toml file.

    This function can either use a provided version from command line arguments,
    or generate a new version based on the current version plus timestamp.

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
      (will generate version based on current version + timestamp)
    """
    cargo_path = Path(__file__).parents[1] / "py-phylo2vec" / "Cargo.toml"

    try:
        if len(sys.argv) > 1:
            # Use provided version
            new_version = sys.argv[1]
        else:
            # Generate version with timestamp
            current_version = get_current_version(cargo_path)
            new_version = generate_version_with_timestamp(current_version)
            print(f"Generated new version: {new_version}")

        # Update the version in the Cargo.toml file
        update_cargo_version(cargo_path, new_version)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
