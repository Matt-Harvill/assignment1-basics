from collections import Counter
import os
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from functools import partial
from .utils import split_text_on_special_tokens


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk_ordered(
    chunk: str, special_tokens: list[str]
) -> tuple[list[list[tuple[bytes, ...]]], list[str], bool]:
    """
    Pre-tokenize a chunk of text, preserving the order of the tokens, and return the special tokens found in the chunk (and their order)
    for reconstruction later. (For encoding)
    """
    # Pre-tokenization pattern
    regex_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Cut out the special tokens first - escape regex metacharacters for safety
    subchunks, special_tokens_found, special_tokens_first = split_text_on_special_tokens(chunk, special_tokens)

    # Run pre-tokenization on each subchunk and store counts for each pre-token across subchunks
    byte_tuples: list[list[tuple[bytes, ...]]] = []
    for subchunk in subchunks:
        byte_tuples_subchunk: list[tuple[bytes, ...]] = []
        for match in re.finditer(regex_pattern, subchunk):
            matched_str = match.group()
            matched_str_bytes = matched_str.encode("utf-8")
            matched_str_bytes_tuple = tuple(matched_str_bytes[i : i + 1] for i in range(len(matched_str_bytes)))
            byte_tuples_subchunk.append(matched_str_bytes_tuple)
        byte_tuples.append(byte_tuples_subchunk)

    return byte_tuples, special_tokens_found, special_tokens_first


def pre_tokenize_chunk_counts(chunk: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """
    Pre-tokenize a chunk of text to get the counts of each pre-token. (For training)
    """
    # Pre-tokenization pattern
    regex_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Cut out the special tokens first - escape regex metacharacters for safety
    subchunks, _, _ = split_text_on_special_tokens(chunk, special_tokens)

    # Run pre-tokenization on each subchunk and store counts for each pre-token across subchunks
    byte_counts: Counter[tuple[bytes, ...]] = Counter()
    for subchunk in subchunks:
        for match in re.finditer(regex_pattern, subchunk):
            matched_str = match.group()
            matched_str_bytes = matched_str.encode("utf-8")
            matched_str_bytes_tuple = tuple(matched_str_bytes[i : i + 1] for i in range(len(matched_str_bytes)))
            byte_counts[matched_str_bytes_tuple] += 1

    return byte_counts


def read_and_chunk_file(input_path: str | os.PathLike, num_desired_chunks: int) -> list[str]:
    """
    Read a text file and chunk it for parallel processing.
    If num_desired_chunks == 1, returns the entire file as a single chunk.
    Otherwise, uses find_chunk_boundaries to split the file.
    """
    with open(input_path, "rb") as f:
        if num_desired_chunks == 1:
            # Read entire file as single chunk
            chunk = f.read().decode("utf-8", errors="ignore")
            return [chunk]
        else:
            # Find chunk boundaries to split for efficient parallelization
            special_token_bytes: bytes = b"<|endoftext|>"
            boundaries = find_chunk_boundaries(
                file=f, desired_num_chunks=num_desired_chunks, split_special_token=special_token_bytes
            )
            chunks: list[str] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
            return chunks


def run_pre_tokenize_parallel(
    chunks: list[str], special_tokens: list[str], num_processes: int
) -> Counter[tuple[bytes, ...]]:
    """
    Run pre_tokenize_chunk_counts in parallel across the given chunks.
    """
    with mp.Pool(num_processes) as pool:
        # Count bytes counts per process
        _partial = partial(pre_tokenize_chunk_counts, special_tokens=special_tokens)
        pre_tokenized_counters: list[Counter[tuple[bytes, ...]]] = pool.map(_partial, chunks)

        # Aggregate counts across processes
        pre_tokenization_counts: Counter[tuple[bytes, ...]] = sum(pre_tokenized_counters, Counter())

    return pre_tokenization_counts


def pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str], num_desired_processes: int | None = None
) -> Counter[tuple[bytes, ...]]:
    """
    Parallel implementation of pre-tokenization for Byte-Pair Encoding (BPE) tokenizer
    - Uses up to all processes if desired
    """

    num_processes = min(mp.cpu_count(), num_desired_processes) if num_desired_processes is not None else mp.cpu_count()

    # Read and chunk the file
    chunks = read_and_chunk_file(input_path, num_desired_chunks=num_processes)

    # Run pre-tokenization in parallel
    pre_tokenization_counts = run_pre_tokenize_parallel(chunks, special_tokens, num_processes)

    return pre_tokenization_counts


if __name__ == "__main__":
    text = "<TOKEN1>Hello, world! This is a test. Hello<TOKEN1><TOKEN1> is a test.<TOKEN1> Hello<TOKEN1><TOKEN2>World"
    special_tokens = ["<TOKEN1>", "<TOKEN2>"]
    a, b, c = split_text_on_special_tokens(text, special_tokens)
    print(a)
    print(b)
    print(c)

    assert abs(len(a) - len(b)) <= 1, "Number of segments and special tokens should differ by at most 1"

    text_segments, special_tokens_found, special_tokens_first = pre_tokenize_chunk_ordered(text, special_tokens)
    print(text_segments)
    print(special_tokens_found)
    print(special_tokens_first)
