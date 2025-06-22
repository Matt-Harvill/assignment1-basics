import time
import random
import string
import os
from .tokenizer import Tokenizer


def get_tokenizer_compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> float:
    """
    Calculate the compression ratio (bytes/token) for a tokenizer on a set of documents.

    Args:
        tokenizer: Tokenizer to evaluate
        documents: List of documents to tokenize

    Returns:
        Compression ratio as bytes per token
    """
    total_bytes = 0
    total_tokens = 0

    for doc in documents:
        # Count bytes in original text
        total_bytes += len(doc.encode("utf-8"))

        # Count tokens after tokenization
        token_ids = tokenizer.encode(doc)
        total_tokens += len(token_ids)

    if total_tokens == 0:
        return float("inf")

    return total_bytes / total_tokens


def get_tokenizer_throughput(tokenizer: Tokenizer, test_size_mb: float = 10.0, use_random_bytes: bool = False) -> float:
    """
    Estimate the throughput of a tokenizer in bytes/second.

    Args:
        tokenizer: Tokenizer to test
        test_size_mb: Size of test data in MB
        use_random_bytes: If True, generate fully random bytes instead of text.

    Returns:
        Throughput in bytes/second
    """
    # Generate random text of specified size
    test_size_bytes = int(test_size_mb * 1024 * 1024)

    if use_random_bytes:
        if test_size_bytes == 0:
            return float("inf")
        random_data = os.urandom(test_size_bytes)
        # Decode into a string, replacing invalid UTF-8 sequences, as the tokenizer expects a string.
        test_text = random_data.decode("utf-8", errors="replace")
    else:
        # Generate random text with realistic character distribution
        chars = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        test_text = "".join(random.choices(chars, k=test_size_bytes))

    # We use the original `test_size_bytes` for the throughput calculation
    # to measure performance against the raw input byte size.

    # Split into chunks to avoid memory issues
    chunk_size = 10000  # 10KB chunks
    chunks = [test_text[i : i + chunk_size] for i in range(0, len(test_text), chunk_size)]

    # Time the tokenization
    start_time = time.time()

    total_tokens = 0
    for chunk in chunks:
        token_ids = tokenizer.encode(chunk)
        total_tokens += len(token_ids)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time == 0:
        return float("inf")

    # Calculate throughput
    throughput = test_size_bytes / elapsed_time

    print(f"Tokenized {test_size_mb:.2f}MB in {elapsed_time:.2f}s")
    print(f"Generated {total_tokens:,} tokens")
    print(f"Throughput: {throughput:,.0f} bytes/s")

    return throughput


def estimate_pile_tokenization_time(
    tokenizer: Tokenizer, pile_size_gb: float = 825.0, throughput_bps: float | None = None
) -> float:
    """
    Estimate how long it would take to tokenize the Pile dataset.

    Args:
        tokenizer: Tokenizer to use
        pile_size_gb: Size of Pile dataset in GB
        throughput_bps: Optional pre-calculated throughput in bytes/second.

    Returns:
        Estimated time in hours
    """
    if throughput_bps is None:
        # If throughput isn't provided, calculate it now.
        throughput_bps = get_tokenizer_throughput(tokenizer)

    if throughput_bps == 0:
        return float("inf")

    pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024

    time_seconds = pile_size_bytes / throughput_bps
    time_hours = time_seconds / 3600

    return time_hours


def analyze_tokenizer_performance(tokenizer: Tokenizer, documents: list[str]) -> dict:
    """
    Comprehensive analysis of tokenizer performance.

    Args:
        tokenizer: Tokenizer to analyze
        documents: List of documents to test on

    Returns:
        Dictionary with performance metrics
    """
    # Compression ratio
    compression_ratio = get_tokenizer_compression_ratio(tokenizer, documents)

    # Throughput
    throughput = get_tokenizer_throughput(tokenizer)

    # Pile tokenization time estimate
    pile_time_hours = estimate_pile_tokenization_time(tokenizer)

    # Token distribution analysis
    all_token_ids = []
    for doc in documents:
        token_ids = tokenizer.encode(doc)
        all_token_ids.extend(token_ids)

    vocab_size = len(tokenizer.vocab)
    unique_tokens_used = len(set(all_token_ids))
    token_coverage = unique_tokens_used / vocab_size if vocab_size > 0 else 0

    return {
        "compression_ratio": compression_ratio,
        "throughput_bytes_per_sec": throughput,
        "pile_tokenization_time_hours": pile_time_hours,
        "total_tokens": len(all_token_ids),
        "unique_tokens_used": unique_tokens_used,
        "vocab_size": vocab_size,
        "token_coverage": token_coverage,
    }
