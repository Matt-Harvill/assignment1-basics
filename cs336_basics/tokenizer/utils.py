import json
from pathlib import Path


def load_vocab_and_merges(vocab_filepath: str | Path, merges_filepath: str | Path) -> tuple[dict[int, str], list[str]]:
    """
    Load vocab.json and merges.txt files from the given file paths.

    Args:
        vocab_filepath: Path to the vocab.json file
        merges_filepath: Path to the merges.txt file

    Returns:
        Tuple of (vocab_dict, merges_list). merges_list will be None if merges.txt doesn't exist.

    Raises:
        FileNotFoundError: If the vocab file doesn't exist.
        FileNotFoundError: If the merges file doesn't exist.
    """
    vocab_path = Path(vocab_filepath)
    merges_path = Path(merges_filepath)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    # Load vocab
    with open(vocab_path) as f:
        vocab = json.load(f)

    # Load merges if it exists
    merges = None
    if merges_path.exists():
        with open(merges_path) as f:
            merges = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Merges file not found: {merges_path}")
    if merges is None:
        raise ValueError("Merges file is improperly formatted")
    if len(merges) == 0:
        raise ValueError("Merges file is empty")

    return vocab, merges


def convert_vocab_to_bytes(vocab: dict[int, str]) -> dict[int, bytes]:
    """
    Convert vocab values from strings to bytes.

    Args:
        vocab: Dictionary mapping token IDs to string tokens

    Returns:
        Dictionary mapping token IDs to bytes tokens
    """
    return {k: v.encode("utf-8") for k, v in vocab.items()}


def convert_merges_to_bytes(merges: list[str] | None) -> list[tuple[bytes, bytes]]:
    """
    Convert merges from strings to bytes tuples.

    Args:
        merges: List of merge strings in format "token1 token2" or None

    Returns:
        List of tuples of bytes representing merges
    """
    if merges is None:
        return []

    merges_bytes = []
    for merge in merges:
        parts = merge.split()
        if len(parts) == 2:
            merges_bytes.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

    return merges_bytes
