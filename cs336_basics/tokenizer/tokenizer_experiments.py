#!/usr/bin/env python3
"""
Script to run tokenizer experiments and answer the 4 questions in the assignment.
"""

from pathlib import Path
from ..utils.paths import get_artifacts_path
from .tokenizer import Tokenizer
from .utils import sample_n_docs_from_file, tokenize_and_serialize_dataset
from .metrics import get_tokenizer_compression_ratio, get_tokenizer_throughput, estimate_pile_tokenization_time


def load_tokenizers():
    """Load the TinyStories and OpenWebText tokenizers."""
    artifacts_path = get_artifacts_path()

    # TinyStories tokenizer (10K vocab)
    tinystories_vocab = artifacts_path / "tokenizers" / "TinyStoriesV2-GPT4-train" / "vocab.json"
    tinystories_merges = artifacts_path / "tokenizers" / "TinyStoriesV2-GPT4-train" / "merges.txt"
    tinystories_tokenizer = Tokenizer.from_files(
        tinystories_vocab, tinystories_merges, special_tokens=["<|endoftext|>"]
    )

    # OpenWebText tokenizer (32K vocab) - assuming it exists
    owt_vocab = artifacts_path / "tokenizers" / "OpenWebText-32K" / "vocab.json"
    owt_merges = artifacts_path / "tokenizers" / "OpenWebText-32K" / "merges.txt"

    if owt_vocab.exists() and owt_merges.exists():
        owt_tokenizer = Tokenizer.from_files(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])
    else:
        print("Warning: OpenWebText tokenizer not found, using TinyStories as fallback")
        owt_tokenizer = tinystories_tokenizer

    return tinystories_tokenizer, owt_tokenizer


def question_a_compression_ratios():
    """Answer question (a): Compression ratios for both tokenizers."""
    print("=" * 60)
    print("Question (a): Compression Ratios")
    print("=" * 60)

    # Load tokenizers
    tinystories_tokenizer, owt_tokenizer = load_tokenizers()

    # Sample documents from both datasets
    data_path = Path("data")

    # Sample from TinyStories
    tinystories_docs = sample_n_docs_from_file(data_path / "TinyStoriesV2-GPT4-train.txt", n=10)

    # Sample from OpenWebText (using condensed version for speed)
    owt_docs = sample_n_docs_from_file(data_path / "owt_train_condensed.txt", n=10)

    # Calculate compression ratios
    ts_compression = get_tokenizer_compression_ratio(tinystories_tokenizer, tinystories_docs)
    owt_compression = get_tokenizer_compression_ratio(owt_tokenizer, owt_docs)

    print(f"TinyStories tokenizer compression ratio: {ts_compression:.2f} bytes/token")
    print(f"OpenWebText tokenizer compression ratio: {owt_compression:.2f} bytes/token")

    return ts_compression, owt_compression


def question_b_cross_tokenization():
    """Answer question (b): What happens when using wrong tokenizer."""
    print("\n" + "=" * 60)
    print("Question (b): Cross-Tokenization Analysis")
    print("=" * 60)

    # Load tokenizers
    tinystories_tokenizer, owt_tokenizer = load_tokenizers()

    # Sample OpenWebText documents
    data_path = Path("data")
    owt_docs = sample_n_docs_from_file(data_path / "owt_train_condensed.txt", n=10)

    # Tokenize with both tokenizers
    owt_with_owt = get_tokenizer_compression_ratio(owt_tokenizer, owt_docs)
    owt_with_ts = get_tokenizer_compression_ratio(tinystories_tokenizer, owt_docs)

    print(f"OpenWebText docs with OpenWebText tokenizer: {owt_with_owt:.2f} bytes/token")
    print(f"OpenWebText docs with TinyStories tokenizer: {owt_with_ts:.2f} bytes/token")
    print(f"Compression ratio difference: {owt_with_ts/owt_with_owt:.2f}x worse")

    return owt_with_owt, owt_with_ts


def question_c_throughput():
    """Answer question (c): Tokenizer throughput and Pile dataset time."""
    print("\n" + "=" * 60)
    print("Question (c): Throughput Analysis")
    print("=" * 60)

    # Load tokenizers
    tinystories_tokenizer, owt_tokenizer = load_tokenizers()

    # Measure throughput for both tokenizers
    print("Measuring TinyStories tokenizer throughput...")
    ts_throughput = get_tokenizer_throughput(tinystories_tokenizer, test_size_mb=10.0)

    print("\nMeasuring OpenWebText tokenizer throughput...")
    owt_throughput = get_tokenizer_throughput(owt_tokenizer, test_size_mb=10.0)

    # Estimate Pile tokenization time
    ts_pile_time = estimate_pile_tokenization_time(tinystories_tokenizer, throughput_bps=ts_throughput)
    owt_pile_time = estimate_pile_tokenization_time(owt_tokenizer, throughput_bps=owt_throughput)

    print(f"\nTinyStories tokenizer throughput: {ts_throughput:,.0f} bytes/s")
    print(f"OpenWebText tokenizer throughput: {owt_throughput:,.0f} bytes/s")
    print("Pile dataset (825GB) tokenization time:")
    print(f"  - TinyStories: {ts_pile_time:.1f} hours")
    print(f"  - OpenWebText: {owt_pile_time:.1f} hours")

    return ts_throughput, owt_throughput, ts_pile_time, owt_pile_time


def question_d_serialize_datasets():
    """Answer question (d): Serialize datasets and explain uint16 choice."""
    print("\n" + "=" * 60)
    print("Question (d): Dataset Serialization")
    print("=" * 60)

    # Load tokenizers
    tinystories_tokenizer, owt_tokenizer = load_tokenizers()

    # Serialize datasets
    data_path = Path("data")
    output_path = Path("cs336_basics/artifacts/tokenized_datasets")

    print("Serializing TinyStories dataset...")
    tokenize_and_serialize_dataset(
        tinystories_tokenizer, data_path / "TinyStoriesV2-GPT4-train.txt", output_path / "tinystories_train.npy"
    )

    # print("Serializing TinyStories validation dataset...")
    # tokenize_and_serialize_dataset(
    #     tinystories_tokenizer,
    #     data_path / "TinyStoriesV2-GPT4-valid.txt",
    #     output_path / "tinystories_valid.npy"
    # )

    # print("Serializing OpenWebText dataset...")
    # tokenize_and_serialize_dataset(
    #     owt_tokenizer,
    #     data_path / "owt_train_condensed.txt",
    #     output_path / "owt_train.npy"
    # )

    # print("Serializing OpenWebText validation dataset...")
    # tokenize_and_serialize_dataset(
    #     owt_tokenizer,
    #     data_path / "owt_valid.txt",
    #     output_path / "owt_valid.npy"
    # )

    # Explain uint16 choice
    print("\nWhy uint16 is appropriate:")
    print("- uint16 can represent values 0-65,535")
    print("- TinyStories vocab size: ~10,000 tokens")
    print("- OpenWebText vocab size: ~32,000 tokens")
    print("- Both fit comfortably within uint16 range")
    print("- uint16 uses 2 bytes per token ID, saving memory compared to uint32")
    print("- Provides good balance between range and memory efficiency")


def main():
    """Run all experiments and answer all questions."""
    print("Running Tokenizer Experiments")
    print("=" * 60)

    # Run all experiments
    # question_a_compression_ratios()
    # question_b_cross_tokenization()
    # question_c_throughput()
    question_d_serialize_datasets()

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
