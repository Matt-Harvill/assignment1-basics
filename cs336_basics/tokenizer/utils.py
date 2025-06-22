import json
from pathlib import Path
import re
import numpy as np
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


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


def sample_n_docs_from_file(file_path: str | Path, n: int, delimiter_token: str = "<|endoftext|>") -> list[str]:
    """
    Sample n documents from a file using the specified delimiter token.

    Args:
        file_path: Path to the file containing documents
        n: Number of documents to sample
        delimiter_token: Token that separates documents

    Returns:
        List of n sampled documents
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the entire file and split by delimiter
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    documents = content.split(delimiter_token)
    # Remove empty documents
    documents = [doc.strip() for doc in documents if doc.strip()]

    if len(documents) < n:
        print(f"Warning: Only {len(documents)} documents available, returning all of them")
        return documents

    # Sample n documents randomly
    return random.sample(documents, n)


def tokenize_and_serialize_dataset(
    tokenizer: "Tokenizer",
    file_path: str | Path,
    output_path: str | Path,
    delimiter_token: str = "<|endoftext|>",
    dtype: type = np.uint16,
) -> None:
    """
    Tokenize a dataset and serialize the token IDs to a NumPy array.
    This version is optimized to handle large files by streaming.
    """
    file_path = Path(file_path)
    output_path = Path(output_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_token_ids = []
    buffer = ""
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(file_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Add chunk to buffer and split by delimiter
            text_to_process = buffer + chunk
            documents = text_to_process.split(delimiter_token)

            # The last part might be an incomplete document, so we save it for the next chunk
            buffer = documents.pop()

            for doc in documents:
                if doc.strip():
                    all_token_ids.extend(tokenizer.encode(doc))

    # Process any remaining text in the buffer
    if buffer.strip():
        all_token_ids.extend(tokenizer.encode(buffer))

    # Convert to NumPy array and save
    token_array = np.array(all_token_ids, dtype=dtype)
    np.save(output_path, token_array)

    print(f"Tokenized dataset into {len(token_array)} tokens")
    print(f"Saved to {output_path}")


def load_serialized_dataset(file_path: str | Path) -> np.ndarray:
    """
    Load a serialized tokenized dataset.

    Args:
        file_path: Path to the .npy file

    Returns:
        NumPy array of token IDs
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return np.load(file_path)


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
            # Strip quotes from the tokens if they exist
            first_token = parts[0].strip('"')
            second_token = parts[1].strip('"')
            merges_bytes.append((string_to_byte(first_token), string_to_byte(second_token)))

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

    # Escape special regex characters in special tokens (in reverse order to match longest tokens first)
    pattern = r"|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))

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
