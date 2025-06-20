from dataclasses import dataclass
from cs336_basics.pretokenization_example import pre_tokenize_dataset_bpe
from collections import Counter


@dataclass
class BPEOutput:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def tokenize_dataset_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPEOutput:
    """
        Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
    tokenizer. Your BPE training function should handle (at least) the following input parameters:
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.
    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    To test your BPE training function against our provided tests, you will first need to implement the
    test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.
    Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
    you can implement the key parts of your training method using some systems language, for instance
    C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
    require copying vs reading directly from Python memory, and make sure to leave build instructions, or
    make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
    in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
    reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
    even faster.
    9
    """

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # Pretokenization
    bytes_counts: Counter[tuple[bytes, ...]] = pre_tokenize_dataset_bpe(
        input_path=input_path, special_tokens=special_tokens
    )

    # Count all byte pairs
    byte_pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for bytes_tuple, count in bytes_counts.items():
        # Get 2 bytes at a time
        for i in range(len(bytes_tuple) - 1):
            byte_pair_counts[(bytes_tuple[i], bytes_tuple[i + 1])] += count

    print(f"bytes_counts: {bytes_counts}\n\n")
    print(f"byte_pair_counts: {byte_pair_counts}\n\n")

    # # Before adding to vocab and merging, we want to sort the counts
    # for key, count in byte_pair_counts.items():
    #     if count not in sorted_byte_pair_counts:
    #         sorted_byte_pair_counts[count] = []
    #     sorted_byte_pair_counts[count].append(key)

    # Now finally start adding bytes to the vocab
    # Add special tokens first
    vocab_index = 0
    for special_token in special_tokens:
        vocab[vocab_index] = special_token.encode("utf-8")

    # # Keep merging until we've filled the vocabulary
    # while vocab_index < vocab_size or len(sorted_byte_pair_counts) == 0:
    #     count, bytes_tuple = sorted_byte_pair_counts.peekitem()

    #     if len(list_of_bytes) == 1:
    #         merges.append()

    return BPEOutput(vocab, merges)


if __name__ == "__main__":
    # Test the tokenize function
    import cProfile
    import pstats

    # # Profile and save results
    # cProfile.run(
    #     'tokenize_dataset_bpe(input_path="/home/matthew/Code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", vocab_size=0, special_tokens=["<|endoftext|>"])',
    #     "profile_stats",
    # )

    # Profile and save results
    cProfile.run(
        'tokenize_dataset_bpe(input_path="/home/matthew/Code/assignment1-basics/data/test.txt", vocab_size=0, special_tokens=["<|endoftext|>"])',
        "profile_stats",
    )

    # Analyze results
    stats = pstats.Stats("profile_stats")
    stats.sort_stats("cumulative")
    stats.print_stats(10)  # Top 10 functions
