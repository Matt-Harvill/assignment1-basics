from pathlib import Path


def get_package_root() -> Path:
    """
    Get the path to the cs336_basics package root directory.

    Returns:
        Path: The absolute path to the package root directory.

    Example:
        >>> from cs336_basics.utils import get_package_root
        >>> root = get_package_root()
        >>> artifacts_dir = root / "artifacts" / "tokenizer"
    """
    # Get the path to this utils module, then go up to the package root
    return Path(__file__).parent.parent


def get_artifacts_path() -> Path:
    """
    Get the path to the artifacts directory.

    Returns:
        Path: The absolute path to the artifacts directory.

    Example:
        >>> from cs336_basics.utils import get_artifacts_path
        >>> artifacts = get_artifacts_path()
        >>> tokenizer_dir = artifacts / "tokenizer"
    """
    return get_package_root() / "artifacts"
