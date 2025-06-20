from .pretokenize import pre_train_tokenizer
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


def train_tokenizer(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = mp.cpu_count()
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Tokenizes the dataset using Byte-Pair Encoding (BPE)

    Args:
        input_path (str | os.PathLike): Path to the input text file
        vocab_size (int): The maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
        num_processes (int): Number of processes to use for pretokenization. Defaults to the number of CPU cores.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: The tokenizer vocabulary and merges
    """

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # Pretokenization
    bytes_counts: dict[tuple[bytes, ...], int] = pre_train_tokenizer(
        input_path=input_path, special_tokens=special_tokens, num_desired_processes=num_processes
    )

    def count_bytes_pairs(bytes_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        bytes_pair_counts: dict[tuple[bytes, bytes], int] = {}
        for bytes_tuple, count in bytes_counts.items():
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + count
        return bytes_pair_counts

    def count_bytes_pairs_and_tuples_present(
        bytes_counts: dict[tuple[bytes, ...], int]
    ) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
        bytes_pair_counts: dict[tuple[bytes, bytes], int] = {}
        bytes_pair_tuples_present: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for bytes_tuple, count in bytes_counts.items():
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + count
                bytes_pair_tuples_present[pair].add(bytes_tuple)
        return bytes_pair_counts, bytes_pair_tuples_present

    # Count all byte pairs
    bytes_pair_counts, bytes_pair_tuples_present = count_bytes_pairs_and_tuples_present(bytes_counts=bytes_counts)

    # Now finally start adding bytes to the vocab
    # Add special tokens first
    vocab_index = 0
    for special_token in special_tokens:
        vocab[vocab_index] = special_token.encode("utf-8")
        vocab_index += 1

    # Add the individual bytes to the vocab
    for i in range(256):
        vocab[vocab_index] = bytes([i])
        vocab_index += 1

    # Helper function for merging to find the most frequent pair this iteration
    def get_most_freq_pair() -> tuple[bytes, bytes]:
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

    # Merge a pair of bytes (handling the bytes_counts and bytes_pair_counts updates)
    def merge_pair(most_freq_pair: tuple[bytes, bytes]):
        nonlocal bytes_pair_counts, bytes_counts, merges, bytes_pair_tuples_present

        bytes_tuples_to_delete: list[tuple[bytes, ...]] = []
        bytes_tuples_to_add: list[tuple[bytes, ...]] = []
        counts_to_add: list[int] = []

        # Create a copy of the set to iterate over safely
        bytes_tuples_to_process = list(bytes_pair_tuples_present[most_freq_pair])

        for bytes_tuple in bytes_tuples_to_process:
            count = bytes_counts[bytes_tuple]

            # Create the new tuple with merge(s)
            new_bytes_list: list[bytes] = []
            i = 0
            while i < len(bytes_tuple) - 1:
                current_pair = (bytes_tuple[i], bytes_tuple[i + 1])
                if current_pair == most_freq_pair:
                    new_bytes_list.append(
                        bytes_tuple[i] + bytes_tuple[i + 1]
                    )  # Join the bytes together into single bytes object
                    i += 2  # Skip the next byte since we merged it
                else:
                    new_bytes_list.append(bytes_tuple[i])
                    i += 1
            # Add the last byte if we haven't processed it
            if i < len(bytes_tuple):
                new_bytes_list.append(bytes_tuple[i])
            new_bytes_tuple: tuple[bytes, ...] = tuple(new_bytes_list)

            # Recount - manually handle dict arithmetic
            before_merge_bytes_pair_counts = count_bytes_pairs({bytes_tuple: count})
            after_merge_bytes_pair_counts = count_bytes_pairs({new_bytes_tuple: count})

            # Clear the old bytes pair counts and tuples present
            for pair, pair_count in before_merge_bytes_pair_counts.items():
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) - pair_count
                if bytes_pair_counts[pair] <= 0:
                    del bytes_pair_counts[pair]
                bytes_pair_tuples_present[pair].remove(bytes_tuple)
                if len(bytes_pair_tuples_present[pair]) == 0:
                    del bytes_pair_tuples_present[pair]

            # Update each final bytes pair count and add the new bytes tuple to bytes_pair_tuples_present
            for pair, pair_count in after_merge_bytes_pair_counts.items():
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + pair_count
                bytes_pair_tuples_present[pair].add(new_bytes_tuple)

            # Store for later processing
            bytes_tuples_to_delete.append(bytes_tuple)
            bytes_tuples_to_add.append(new_bytes_tuple)
            counts_to_add.append(count)

        # Delete the old bytes_tuples and add the new ones after iteration is complete
        for bytes_tuple in bytes_tuples_to_delete:
            del bytes_counts[bytes_tuple]
        for new_bytes_tuple, tuple_count in zip(bytes_tuples_to_add, counts_to_add):
            bytes_counts[new_bytes_tuple] = bytes_counts.get(new_bytes_tuple, 0) + tuple_count

        # Add the merged pair to merges
        merges.append(most_freq_pair)

    # Add all other tokens until we've filled the vocabulary
    while vocab_index < vocab_size and len(bytes_pair_counts) > 0:
        # Find the most frequent byte pair
        most_freq_pair = get_most_freq_pair()

        # Add to vocabulary
        merged_bytes = most_freq_pair[0] + most_freq_pair[1]
        vocab[vocab_index] = merged_bytes
        vocab_index += 1

        # Merge it
        merge_pair(most_freq_pair)

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
