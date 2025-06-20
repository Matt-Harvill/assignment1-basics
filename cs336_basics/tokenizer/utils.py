import json
from pathlib import Path


def load_vocab_and_merges(
    vocab_filepath: str | Path, merges_filepath: str | Path
) -> tuple[dict[int, str], list[str] | None]:
    """
    Load vocab.json and merges.txt files from the given file paths.

    Args:
        vocab_filepath: Path to the vocab.json file
        merges_filepath: Path to the merges.txt file

    Returns:
        Tuple of (vocab_dict, merges_list). merges_list will be None if merges.txt doesn't exist.

    Raises:
        FileNotFoundError: If the vocab file doesn't exist.
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

    return vocab, merges
