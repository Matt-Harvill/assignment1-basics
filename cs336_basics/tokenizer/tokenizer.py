from collections.abc import Iterable, Iterator
from .utils import load_vocab_and_merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)

        # Convert vocab values from strings to bytes
        vocab_bytes = {k: v.encode("utf-8") for k, v in vocab.items()}

        # Convert merges from strings to bytes tuples@
        merges_bytes = []
        if merges is not None:
            for merge in merges:
                parts = merge.split()
                if len(parts) == 2:
                    merges_bytes.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab_bytes, merges_bytes, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
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
