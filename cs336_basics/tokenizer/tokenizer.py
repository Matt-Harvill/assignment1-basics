from collections.abc import Iterable, Iterator
from cs336_basics.utils.paths import get_artifacts_path
from .utils import load_vocab_and_merges, convert_vocab_to_bytes, convert_merges_to_bytes
from .pretokenizer import pre_tokenize_chunk_ordered
from pathlib import Path


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # --- OPTIMIZATION ---
        # Create a dictionary mapping merge pairs to their rank (priority) for fast lookups.
        # Lower rank means higher priority.
        self.ranks = {merge: i for i, merge in enumerate(self.merges)}
        # --- END OPTIMIZATION ---

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str | Path, merges_filepath: str | Path, special_tokens=None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab_strs, merges_strs = load_vocab_and_merges(vocab_filepath, merges_filepath)
        vocab = convert_vocab_to_bytes(vocab_strs)
        merges = convert_merges_to_bytes(merges_strs)
        return cls(vocab, merges, special_tokens)

    def _get_pairs(self, parts: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Helper function to get all adjacent pairs from a list of tokens."""
        return set(zip(parts, parts[1:]))

    def _bpe_merge(self, segment: list[bytes]) -> list[bytes]:
        """
        Repeatedly merges the highest-priority pair in a list of byte chunks.
        This is the core of the optimized BPE algorithm.
        """
        parts = list(segment)
        while True:
            pairs = self._get_pairs(parts)
            if not pairs:
                break

            # Find the highest-priority pair (lowest rank) in the current text
            inf = float("inf")
            best_pair = min(pairs, key=lambda pair: self.ranks.get(pair, inf))

            if self.ranks.get(best_pair, inf) == inf:
                break  # No more merges possible

            # Merge the best pair
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # First we pre-tokenize the text (saving positional information about the special tokens)
        text_segments_bytes, special_tokens, special_tokens_first = pre_tokenize_chunk_ordered(
            text, [] if self.special_tokens is None else self.special_tokens
        )

        # --- OPTIMIZATION ---
        # Apply BPE merges to each segment using the fast algorithm
        final_merged_segments = []
        for segment in text_segments_bytes:
            # Flatten the nested structure from the pre-tokenizer into a single list of bytes
            merged_segment = []
            for word_bytes_tuple in segment:
                merged_word = self._bpe_merge(list(word_bytes_tuple))
                merged_segment.extend(merged_word)
            final_merged_segments.append(merged_segment)
        # --- END OPTIMIZATION ---

        # After we've merged all the bytes, we need to convert the bytes objects to token ids
        token_ids: list[list[int]] = []
        for text_segment_subset_bytes in final_merged_segments:
            token_ids_subset: list[int] = []
            # Note: the original code had a bug here. `text_segment_subset_bytes` is the final list of merged tokens.
            for bytes_object in text_segment_subset_bytes:
                if bytes_object in self.inverse_vocab:
                    token_id = self.inverse_vocab[bytes_object]
                    token_ids_subset.append(token_id)
                else:
                    # Fallback for tokens not in vocab: encode as individual bytes.
                    # This should not happen with a correctly trained BPE tokenizer but adds robustness.
                    for byte in bytes_object:
                        token_id = self.inverse_vocab[bytes([byte])]
                        token_ids_subset.append(token_id)
            token_ids.append(token_ids_subset)

        # Convert special tokens to token ids
        special_tokens_ids = [self.inverse_vocab[special_token.encode("utf-8")] for special_token in special_tokens]

        # Interleave text segments with special tokens
        all_token_ids: list[int] = []
        if special_tokens_first:
            # Handle case where text starts with a special token
            for i in range(len(token_ids)):
                all_token_ids.append(special_tokens_ids[i])
                all_token_ids.extend(token_ids[i])
            if len(special_tokens_ids) > len(token_ids):
                all_token_ids.append(special_tokens_ids[-1])
        else:
            # Handle case where text does not start with a special token
            for i in range(len(token_ids)):
                all_token_ids.extend(token_ids[i])
                if i < len(special_tokens_ids):
                    all_token_ids.append(special_tokens_ids[i])

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """

        ids_to_bytes = [self.vocab[id] for id in ids]
        bytes_to_text = b"".join(ids_to_bytes).decode("utf-8", errors="replace")
        return bytes_to_text


# Test the tokenizer on a few edge cases (small examples)
if __name__ == "__main__":
    # Use argparse to get the vocab and merges files from tokenizer name (and some utils to get the path)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer_name", type=str, default="TinyStoriesV2-GPT4-train")
    args = parser.parse_args()
    artifacts_path = get_artifacts_path()
    vocab_filepath = artifacts_path / "tokenizers" / args.tokenizer_name / "vocab.json"
    merges_filepath = artifacts_path / "tokenizers" / args.tokenizer_name / "merges.txt"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])

    # Test the tokenizer on a few small examples

    test_strings = [
        """<|endoftext|>\n""" "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>",
        "s",
        "🙃",
        "Hello, world!",
        "   Hello, world! This is a test.",
        "Hello, world! This is a test. This is a test. This is a test. This is a test.",
        "Hello, world! <|endoftext|><|endoftext|> this is a simple example<|endoftext|> <|endoftext|> of a sentence",
        "<|endoftext|>Hello, world! <|endoftext|> this is a simple example <|endoftext|> of a sentence<|endoftext|>",
        " <|endoftext|>Hello, world! <|endoftext|> this is a simple example <|endoftext|> of a sentence<|endoftext|><|endoftext|><|endoftext|>",
    ]

    for test_string in test_strings:
        print(f"Token IDs: {tokenizer.encode(test_string)}")
        print(f"Token IDs Iterable: {list(tokenizer.encode_iterable([test_string]))}")
        assert tuple(tokenizer.encode(test_string)) == tuple(
            tokenizer.encode_iterable([test_string])
        ), "Token IDs != Token IDs Iterable"

        print(f"Decoded(Encoded): {tokenizer.decode(tokenizer.encode(test_string))}")
        assert tokenizer.decode(tokenizer.encode(test_string)) == test_string, "Decoded(Encoded) != Test String"
