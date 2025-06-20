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
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

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

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        # First we pre-tokenize the text (saving positional information about the special tokens)
        text_segments_bytes, special_tokens, special_tokens_first = pre_tokenize_chunk_ordered(
            text, [] if self.special_tokens is None else self.special_tokens
        )

        # Then we loop over the merges and apply them to our text segments
        for merge in self.merges:
            for text_segment_subset_bytes in text_segments_bytes:
                for text_segment_bytes in text_segment_subset_bytes:
                    merged_segment_bytes: list[bytes] = []
                    i = 0
                    while i < len(text_segment_bytes) - 1:
                        if text_segment_bytes[i] == merge[0] and text_segment_bytes[i + 1] == merge[1]:
                            # Replace two bytes objects with the concatenated bytes object
                            merged_segment_bytes.append(text_segment_bytes[i] + text_segment_bytes[i + 1])
                            i += 2
                        else:
                            merged_segment_bytes.append(text_segment_bytes[i])
                            i += 1
                    text_segment_bytes = tuple(merged_segment_bytes)

        # After we've merged all the bytes, we need to convert the bytes objects to token ids
        token_ids: list[list[int]] = []
        for text_segment_subset_bytes in text_segments_bytes:
            token_ids_subset: list[int] = []
            for text_segment_bytes in text_segment_subset_bytes:
                for bytes_object in text_segment_bytes:
                    # bytes_object will always be in the vocab
                    token_ids_subset.append(self.inverse_vocab[bytes_object])
            token_ids.append(token_ids_subset)

        # Convert special tokens to token ids
        special_tokens_ids = [self.inverse_vocab[special_token.encode("utf-8")] for special_token in special_tokens]

        # Interleave text segments with special tokens
        all_token_ids: list[int] = []
        for i in range(min(len(token_ids), len(special_tokens_ids))):
            if special_tokens_first:
                all_token_ids.append(special_tokens_ids[i])
            all_token_ids.extend(token_ids[i])
            if not special_tokens_first:
                all_token_ids.append(special_tokens_ids[i])
        if len(token_ids) > len(special_tokens_ids):
            all_token_ids.extend(token_ids[-1])
        if len(token_ids) < len(special_tokens_ids):
            all_token_ids.append(special_tokens_ids[-1])

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        raise NotImplementedError("encode_iterable method not yet implemented")

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        raise NotImplementedError("decode method not yet implemented")


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
    print(tokenizer.encode("Hello, world!"))
    print(tokenizer.encode("Hello, world! This is a test."))
    print(tokenizer.encode("Hello, world! This is a test. This is a test. This is a test. This is a test."))

    # Test the tokenizer on a few edge cases (special tokens)
    print(
        tokenizer.encode(
            "Hello, world! <|endoftext|><|endoftext|> this is a simple example<|endoftext|> <|endoftext|> of a sentence"
        )
    )
    print(
        tokenizer.encode(
            "<|endoftext|>Hello, world! <|endoftext|> this is a simple example <|endoftext|> of a sentence<|endoftext|>"
        )
    )
    print(
        tokenizer.encode(
            " <|endoftext|>Hello, world! <|endoftext|> this is a simple example <|endoftext|> of a sentence<|endoftext|><|endoftext|><|endoftext|>"
        )
    )
