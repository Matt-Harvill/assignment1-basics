from cs336_basics.pretokenization_example import pre_tokenize_dataset_bpe
from collections import Counter
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
    bytes_counts: Counter[tuple[bytes, ...]] = pre_tokenize_dataset_bpe(
        input_path=input_path, special_tokens=special_tokens, num_desired_processes=24
    )

    def count_bytes_pairs(bytes_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
        byte_pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for bytes_tuple, count in bytes_counts.items():
            for i in range(len(bytes_tuple) - 1):
                byte_pair_counts[(bytes_tuple[i], bytes_tuple[i + 1])] += count
        return byte_pair_counts

    # Count all byte pairs
    bytes_pair_counts: Counter[tuple[bytes, bytes]] = count_bytes_pairs(bytes_counts=bytes_counts)

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

    # # Use a SortedDict to get_most_freq_pair quicker
    # sorted_byte_pair_frequency_tracker = SortedBytePairFrequencyTracker(bytes_pair_counts=bytes_pair_counts)
    # def get_most_freq_pair_fast() -> tuple[bytes, bytes]:
    #     return sorted_byte_pair_frequency_tracker.get_most_frequent_byte_pair()

    # def merge_pair_fast(most_freq_pair: tuple[bytes, bytes]):

    #     # We want to replace all instances of the most frequent pair with a single bytes object
    #     # We need to handle cases where this byte pair occurs multiple times in the same bytes tuple
    #     # We will make a list of indices where the most frequent pair occurs
    #     # We will then replace them iteratively and add/subtract the counts from the sorted byte pair frequency tracker
    #     # Then since each pair takes two indices, we will keep a counter of how many indices we have processed to update
    #     # the later pair indices for proper updating

    #     # Get the indices of the most frequent pair
    #     for bytes_tuple, count in bytes_counts.items():
    #         indices_to_replace: list[int] = []
    #         for i in range(len(bytes_tuple) - 1):
    #             if bytes_tuple[i] == most_freq_pair[0] and bytes_tuple[i + 1] == most_freq_pair[1]:
    #                 indices_to_replace.append(i)

    #         # Replace the indices
    #         pairs_replaced: int = 0
    #         for i in range(len(indices_to_replace) - 1):
    #             bytes_tuple[indices_to_replace[i]] = bytes_tuple[indices_to_replace[i]] + bytes_tuple[indices_to_replace[i + 1]]
    #             bytes_tuple[indices_to_replace[i + 1]] = None

    #         # Remove the None values

    #     sorted_byte_pair_frequency_tracker.subtract(byte_pair=most_freq_pair, count=1)
    #     new_bytes_tuple = most_freq_pair[0] + most_freq_pair[1]
    #     sorted_byte_pair_frequency_tracker.add(byte_pair=new_bytes_tuple, count=1)

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

            # Recount
            before_merge_bytes_pair_counts = count_bytes_pairs(Counter({bytes_tuple: count}))
            after_merge_bytes_pair_counts = count_bytes_pairs(Counter({new_bytes_tuple: count}))
            bytes_pair_counts -= before_merge_bytes_pair_counts
            bytes_pair_counts += after_merge_bytes_pair_counts

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
        logger.debug(f"bytes_counts: {bytes_counts}")
        logger.debug(f"byte_pair_counts: {bytes_pair_counts}")

        # Find the most frequent byte pair
        most_freq_pair = get_most_freq_pair()
        logger.debug(f"most_freq_pair: {most_freq_pair}")

        # Add to vocabulary
        merged_bytes = most_freq_pair[0] + most_freq_pair[1]
        vocab[vocab_index] = merged_bytes
        vocab_index += 1

        # Merge it
        merge_pair(most_freq_pair)

    logger.debug(f"Final bytes_counts: {bytes_counts}")
    logger.debug(f"Final byte_pair_counts: {bytes_pair_counts}")

    return (vocab, merges)


if __name__ == "__main__":
    # # Test the tokenize function
    # import cProfile
    # import pstats

    # # Profile and save results
    # cProfile.run(
    #     'tokenize_dataset_bpe(input_path="/home/matthew/Code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", vocab_size=0, special_tokens=["<|endoftext|>"])',
    #     "profile_stats",
    # )

    # # Profile and save results
    # cProfile.run(
    #     'tokenize_dataset_bpe(input_path="/home/matthew/Code/assignment1-basics/data/simple_test.txt", vocab_size=10, special_tokens=["<|endoftext|>"])',
    #     "profile_stats",
    # )

    # # Analyze results
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats("cumulative")
    # stats.print_stats(10)  # Top 10 functions

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
