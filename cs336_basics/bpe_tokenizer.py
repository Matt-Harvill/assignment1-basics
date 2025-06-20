from cs336_basics.pretokenization_example import pre_tokenize_dataset_bpe
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tokenize_dataset_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Tokenizes the dataset using Byte-Pair Encoding (BPE)

    Args:
        input_path (str | os.PathLike): Path to the input text file
        vocab_size (int): The maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: The tokenizer vocabulary and merges
    """

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # Pretokenization
    bytes_counts: dict[tuple[bytes, ...], int] = pre_tokenize_dataset_bpe(
        input_path=input_path, special_tokens=special_tokens, num_desired_processes=24
    )

    def count_bytes_pairs(bytes_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        byte_pair_counts: dict[tuple[bytes, bytes], int] = {}
        for bytes_tuple, count in bytes_counts.items():
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                byte_pair_counts[pair] = byte_pair_counts.get(pair, 0) + count
        return byte_pair_counts

    # Count all byte pairs
    bytes_pair_counts: dict[tuple[bytes, bytes], int] = count_bytes_pairs(bytes_counts=bytes_counts)

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
        nonlocal bytes_pair_counts, bytes_counts, merges

        bytes_tuples_to_delete: list[tuple[bytes, ...]] = []
        bytes_tuples_to_add: list[tuple[bytes, ...]] = []
        counts_to_add: list[int] = []

        for bytes_tuple, count in bytes_counts.items():
            # See if the most_freq_pair in this tuple of bytes
            pair_in_tuple = False
            for i in range(len(bytes_tuple) - 1):
                if most_freq_pair == (bytes_tuple[i], bytes_tuple[i + 1]):
                    pair_in_tuple = True
                    break

            # Only change this tuple if the pair is present
            if not pair_in_tuple:
                continue

            # Create the new tuple with merge(s)
            new_bytes_list: list[bytes] = []
            i = 0
            while i < len(bytes_tuple) - 1:
                if most_freq_pair == (bytes_tuple[i], bytes_tuple[i + 1]):
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

            # Subtract before_merge counts
            for pair, pair_count in before_merge_bytes_pair_counts.items():
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) - pair_count
                # Remove zero or negative counts
                if bytes_pair_counts[pair] <= 0:
                    del bytes_pair_counts[pair]

            # Add after_merge counts
            for pair, pair_count in after_merge_bytes_pair_counts.items():
                bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + pair_count

            # Store for later processing
            bytes_tuples_to_delete.append(bytes_tuple)
            bytes_tuples_to_add.append(new_bytes_tuple)
            counts_to_add.append(count)

        # Delete the old bytes_tuples and add the new ones after iteration is complete
        for bytes_tuple in bytes_tuples_to_delete:
            del bytes_counts[bytes_tuple]
        for new_bytes_tuple, count in zip(bytes_tuples_to_add, counts_to_add):
            bytes_counts[new_bytes_tuple] = bytes_counts.get(new_bytes_tuple, 0) + count

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
    vocab, merges = tokenize_dataset_bpe(
        input_path="/home/matthew/Code/assignment1-basics/data/test.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Print the vocab and merges
    logger.info(f"vocab: {vocab}")
    logger.info(f"merges: {merges}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Number of merges: {len(merges)}")
