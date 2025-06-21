import json
from pathlib import Path
import re


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


def get_single_byte_tokens() -> list[str]:
    """
    Get a list of all single byte tokens.
    """
    return [chr(i + 256) for i in range(256)]


def byte_to_string(byte: bytes) -> str:
    """
    Convert a byte to a string.
    """
    if len(byte) == 1:
        return chr(byte[0] + 256)
    return byte.decode("utf-8", errors="replace")


def string_to_byte(string: str) -> bytes:
    """
    Convert a string to a byte.
    """
    if string in get_single_byte_tokens():
        return bytes([ord(string) - 256])
    return string.encode("utf-8")


def convert_vocab_to_bytes(vocab: dict[int, str]) -> dict[int, bytes]:
    """
    Convert vocab values from strings to bytes.

    Args:
        vocab: Dictionary mapping token IDs to string tokens

    Returns:
        Dictionary mapping token IDs to bytes tokens
    """
    return {k: string_to_byte(v) for k, v in vocab.items()}


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
            merges_bytes.append((string_to_byte(parts[0]), string_to_byte(parts[1])))

    return merges_bytes


def split_text_on_special_tokens(text: str, special_tokens: list[str]) -> tuple[list[str], list[str], bool]:
    """
    Split text on special tokens and return text segments, special tokens, and whether text starts with a special token for reconstruction later

    Returns:
        tuple: (text_segments, special_tokens_found, starts_with_special_token) where:
            - text_segments: List of text segments (may include empty strings)
            - special_tokens_found: List of special tokens found in order of appearance
            - starts_with_special_token: Boolean indicating if text starts with a special token
    """
    if not special_tokens:
        return [text], [], False

    # Escape special regex characters in special tokens
    pattern = r"|".join(map(re.escape, special_tokens))

    # Get text segments by splitting on special tokens
    text_segments = re.split(pattern, text)

    # Get special tokens in order of appearance
    special_tokens_found = re.findall(pattern, text)

    # Check if text starts with a special token
    starts_with_special_token = any(text.startswith(token) for token in special_tokens)

    # If the first segment is empty, remove it (This means that the first special token is at the start of the text)
    if text_segments[0] == "":
        text_segments = text_segments[1:]

    return text_segments, special_tokens_found, starts_with_special_token
