from .pretokenizer import pretokenize
from ..utils.paths import get_artifacts_path
import logging
import os
import argparse
from collections import defaultdict
import multiprocessing as mp
import cProfile
import pstats
import json
from pathlib import Path


def count_bytes_pairs(bytes_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count byte pairs from bytes tuples.

    Args:
        bytes_counts: Dictionary mapping bytes tuples to their counts

    Returns:
        Dictionary mapping byte pairs to their counts
    """
    bytes_pair_counts: dict[tuple[bytes, bytes], int] = {}

    for bytes_tuple, count in bytes_counts.items():
        for i in range(len(bytes_tuple) - 1):
            pair = (bytes_tuple[i], bytes_tuple[i + 1])
            bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + count

    return bytes_pair_counts


def count_bytes_pairs_with_tuples_present(
    bytes_counts: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Since we have a bunch of strings broken into bytes tuples and we want to merge bytes pairs,
    We need to count the number of times each byte pair appears in the bytes tuples.
    For efficiency, we also track which tuples contain each byte pair so we don't need to iterate over all tuples later when merging.

    Args:
        bytes_counts: Dictionary mapping bytes tuples to their counts

    Returns:
        Tuple of (pair_counts, pair_to_tuples_mapping)
    """
    bytes_pair_counts: dict[tuple[bytes, bytes], int] = {}
    bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for bytes_tuple, count in bytes_counts.items():
        for i in range(len(bytes_tuple) - 1):
            pair = (bytes_tuple[i], bytes_tuple[i + 1])
            bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + count
            bytes_pair_tuples_present[pair].add(bytes_tuple)

    return bytes_pair_counts, bytes_pair_tuples_present


def add_special_tokens_to_vocab(vocab: dict[int, bytes], special_tokens: list[str], vocab_index: int) -> int:
    for special_token in special_tokens:
        vocab[vocab_index] = special_token.encode("utf-8")
        vocab_index += 1

    return vocab_index


def add_bytes_to_vocab(vocab: dict[int, bytes], vocab_index: int) -> int:
    for i in range(256):
        vocab[vocab_index] = bytes([i])
        vocab_index += 1

    return vocab_index


def add_merge_to_vocab(vocab: dict[int, bytes], merge: tuple[bytes, bytes], vocab_index: int) -> int:
    vocab[vocab_index] = merge[0] + merge[1]
    vocab_index += 1
    return vocab_index


def get_most_freq_pair(bytes_pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    Find the most frequent pair in the bytes pair counts. If there are multiple pairs with the same highest count,
    return the lexographically highest pair.
    """
    most_freq: int = 0
    most_freq_byte_pairs: list[tuple[bytes, bytes]] = []

    # Get the highest count in the list
    for byte_pair, count in bytes_pair_counts.items():
        if count > most_freq:
            most_freq_byte_pairs = [byte_pair]
            most_freq = count
        elif count == most_freq:
            most_freq_byte_pairs.append(byte_pair)

    # Find the lexographically highest value from the most_freq_byte_pairs
    # Sort based on the raw byte values
    highest_lexographic_pair = max(most_freq_byte_pairs, key=lambda x: (x[0], x[1]))

    return highest_lexographic_pair


def create_new_merged_tuple(bytes_tuple: tuple[bytes, ...], most_freq_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Create a new bytes tuple by merging the most frequent pair."""
    result = []
    i = 0
    while i < len(bytes_tuple) - 1:
        if (bytes_tuple[i], bytes_tuple[i + 1]) == most_freq_pair:
            result.append(bytes_tuple[i] + bytes_tuple[i + 1])
            i += 2
        else:
            result.append(bytes_tuple[i])
            i += 1
    if i < len(bytes_tuple):
        result.append(bytes_tuple[i])
    return tuple(result)


def update_pair_counts_for_tuple(
    bytes_tuple: tuple[bytes, ...],
    new_bytes_tuple: tuple[bytes, ...],
    count: int,
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """Update pair counts when replacing a tuple with its merged version."""
    # Remove old pair counts
    before_merge_counts = count_bytes_pairs({bytes_tuple: count})
    for pair, pair_count in before_merge_counts.items():
        bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) - pair_count
        if bytes_pair_counts[pair] <= 0:
            del bytes_pair_counts[pair]
        bytes_pair_tuples_present[pair].remove(bytes_tuple)
        if len(bytes_pair_tuples_present[pair]) == 0:
            del bytes_pair_tuples_present[pair]

    # Add new pair counts
    after_merge_counts = count_bytes_pairs({new_bytes_tuple: count})
    for pair, pair_count in after_merge_counts.items():
        bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + pair_count
        bytes_pair_tuples_present[pair].add(new_bytes_tuple)


def process_tuple_merge(
    bytes_tuple: tuple[bytes, ...],
    most_freq_pair: tuple[bytes, bytes],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    bytes_counts: dict[tuple[bytes, ...], int],
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    """Process merging of a single tuple and return old and new tuples."""
    count = bytes_counts[bytes_tuple]
    new_bytes_tuple = create_new_merged_tuple(bytes_tuple, most_freq_pair)

    update_pair_counts_for_tuple(bytes_tuple, new_bytes_tuple, count, bytes_pair_counts, bytes_pair_tuples_present)

    return bytes_tuple, new_bytes_tuple


def merge_pair(
    most_freq_pair: tuple[bytes, bytes],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    bytes_counts: dict[tuple[bytes, ...], int],
) -> None:
    """Merge the most frequent pair of bytes this iteration."""
    tuples_to_process = list(bytes_pair_tuples_present[most_freq_pair])
    old_tuples, new_tuples = [], []

    # Process each tuple containing the most frequent pair
    for bytes_tuple in tuples_to_process:
        old_tuple, new_tuple = process_tuple_merge(
            bytes_tuple, most_freq_pair, bytes_pair_counts, bytes_pair_tuples_present, bytes_counts
        )
        old_tuples.append(old_tuple)
        new_tuples.append(new_tuple)

    # Update the main bytes_counts dictionary
    for old_tuple, new_tuple in zip(old_tuples, new_tuples):
        count = bytes_counts[old_tuple]
        del bytes_counts[old_tuple]
        bytes_counts[new_tuple] = bytes_counts.get(new_tuple, 0) + count


def fill_vocab(
    vocab: dict[int, bytes],
    vocab_size: int,
    special_tokens: list[str],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    bytes_counts: dict[tuple[bytes, ...], int],
) -> tuple[int, list[tuple[bytes, bytes]]]:
    """Fill vocabulary by adding special tokens, bytes, and iteratively merging the most frequent pairs."""
    vocab_index: int = 0
    merges: list[tuple[bytes, bytes]] = []

    # Add special tokens and bytes to the vocab
    vocab_index = add_special_tokens_to_vocab(vocab=vocab, special_tokens=special_tokens, vocab_index=vocab_index)
    vocab_index = add_bytes_to_vocab(vocab=vocab, vocab_index=vocab_index)

    while vocab_index < vocab_size and len(bytes_pair_counts) > 0:
        most_freq_pair = get_most_freq_pair(bytes_pair_counts=bytes_pair_counts)
        merge_pair(most_freq_pair, bytes_pair_counts, bytes_pair_tuples_present, bytes_counts)

        vocab_index = add_merge_to_vocab(vocab=vocab, merge=most_freq_pair, vocab_index=vocab_index)
        merges.append(most_freq_pair)

    return vocab_index, merges


def train_tokenizer(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte-Pair Encoding (BPE) tokenizer on a text file.

    Args:
        input_path (str | os.PathLike): Path to the input text file
        vocab_size (int): The maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
        num_processes (int | None): Number of processes to use for pretokenization.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: The tokenizer vocabulary and merges
    """

    vocab: dict[int, bytes] = {}

    # Pretokenize the input text into subchunks (tuples of bytes)
    bytes_counts: dict[tuple[bytes, ...], int] = pretokenize(
        input_path=input_path, special_tokens=special_tokens, num_desired_processes=num_processes
    )

    # Count bytes pairs and track which tuples contain each pair
    bytes_pair_counts, bytes_pair_tuples_present = count_bytes_pairs_with_tuples_present(bytes_counts=bytes_counts)

    # Fill vocabulary by adding initial items and merging pairs
    vocab_index, merges = fill_vocab(
        vocab, vocab_size, special_tokens, bytes_pair_counts, bytes_pair_tuples_present, bytes_counts
    )

    return (vocab, merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file")
    parser.add_argument("-i", "--input_path", help="Path to the input text file", required=True)
    parser.add_argument(
        "-v",
        "--vocab_size",
        type=int,
        help="Maximum vocabulary size (including initial bytes and special tokens)",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to add to vocabulary (default: <|endoftext|>)",
    )
    parser.add_argument(
        "-p",
        "--num-processes",
        type=int,
        default=None,
        required=False,
        help=f"Number of processes to use for pretokenization (default: {mp.cpu_count()})",
    )

    args = parser.parse_args()

    # Profile the tokenization
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("time")
    stats.print_stats(10)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Print the vocab and merges
    logger.debug(f"vocab: {vocab}")
    logger.debug(f"merges: {merges}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Number of merges: {len(merges)}")

    # Save vocab and merges to the artifacts/tokenizers/{dataset_name}/merges.txt and vocab.json
    dataset_name = Path(args.input_path).stem  # Get filename without extension

    # Create the output directory if it doesn't exist
    output_dir = get_artifacts_path() / "tokenizers" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vocab
    with open(output_dir / "vocab.json", "w") as f:
        json.dump({str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f, indent=2)
    logger.info(f"Saved vocab to {output_dir / 'vocab.json'}")

    # Save merges
    with open(output_dir / "merges.txt", "w") as f:
        for merge in merges:
            f.write(
                f"\"{merge[0].decode('utf-8', errors='replace')}\" \"{merge[1].decode('utf-8', errors='replace')}\"\n"
            )
    logger.info(f"Saved merges to {output_dir / 'merges.txt'}")

    logger.info(f"Saved tokenizer to {output_dir}")
