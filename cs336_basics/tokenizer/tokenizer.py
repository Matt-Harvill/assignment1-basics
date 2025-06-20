from collections.abc import Iterable, Iterator

from .utils import load_vocab_and_merges, convert_vocab_to_bytes, convert_merges_to_bytes
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

        # text_segments, special_tokens, special_tokens_first = pre_tokenize_chunk_ordered(text, [] if self.special_tokens is None else self.special_tokens)

        # for text_segment in text_segments:
        #     if text_segment in self.vocab:

        raise NotImplementedError("encode method not yet implemented")

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
