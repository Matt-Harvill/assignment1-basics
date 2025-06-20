from dataclasses import dataclass

from ..utils.paths import get_artifacts_path
from .utils import load_vocab_and_merges


@dataclass
class TokenizerStats:
    """Container for tokenizer statistics."""

    vocab_size: int
    longest_token: str
    longest_token_length: int
    longest_token_id: int
    average_token_length: float
    merges_count: int | None = None
    tokenizer_name: str | None = None


def load_tokenizer_files(tokenizer_name: str) -> tuple[dict[int, str], list | None]:
    """
    Load vocab.json and merges.txt files for a specified tokenizer.

    Args:
        tokenizer_name: Name of the tokenizer directory (e.g., "TinyStoriesV2-GPT4-train")

    Returns:
        Tuple of (vocab_dict, merges_list). merges_list will be None if merges.txt doesn't exist.

    Raises:
        FileNotFoundError: If the tokenizer directory or vocab.json doesn't exist.
    """
    artifacts_path = get_artifacts_path()
    tokenizer_path = artifacts_path / "tokenizers" / tokenizer_name

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    return load_vocab_and_merges(vocab_path, merges_path)


def get_tokenizer_statistics(tokenizer_name: str) -> TokenizerStats:
    """
    Calculate comprehensive statistics for a tokenizer.

    Args:
        tokenizer_name: Name of the tokenizer directory

    Returns:
        TokenizerStats object containing all calculated statistics
    """
    vocab, merges = load_tokenizer_files(tokenizer_name)

    # Calculate basic vocab statistics
    vocab_size = len(vocab)

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_length = len(longest_token)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]

    # Calculate average token length
    total_length = sum(len(token) for token in vocab.values())
    average_token_length = total_length / vocab_size

    # Count merges if available
    merges_count = len(merges) if merges is not None else None

    return TokenizerStats(
        vocab_size=vocab_size,
        longest_token=longest_token,
        longest_token_length=longest_token_length,
        longest_token_id=longest_token_id,
        average_token_length=average_token_length,
        merges_count=merges_count,
        tokenizer_name=tokenizer_name,
    )


def display_tokenizer_statistics(stats: TokenizerStats, verbose: bool = True) -> None:
    """
    Display tokenizer statistics in a formatted way.

    Args:
        stats: TokenizerStats object to display
        verbose: If True, show detailed information including the longest token
    """
    print(f"Tokenizer Statistics: {stats.tokenizer_name}")
    print("=" * 50)
    print(f"Vocabulary size: {stats.vocab_size:,}")
    print(f"Average token length: {stats.average_token_length:.2f} characters")

    if verbose:
        print(f"Longest token: '{stats.longest_token}' (ID: {stats.longest_token_id})")
        print(f"Longest token length: {stats.longest_token_length} characters")

    if stats.merges_count is not None:
        print(f"Number of merges: {stats.merges_count:,}")
    else:
        print("Merges file: Not found")

    print()


def analyze_tokenizer(tokenizer_name: str, verbose: bool = True) -> TokenizerStats:
    """
    Convenience function to load, analyze, and display tokenizer statistics.

    Args:
        tokenizer_name: Name of the tokenizer directory
        verbose: If True, show detailed information

    Returns:
        TokenizerStats object
    """
    stats = get_tokenizer_statistics(tokenizer_name)
    display_tokenizer_statistics(stats, verbose)
    return stats


def list_available_tokenizers() -> list[str]:
    """
    List all available tokenizers in the artifacts directory.

    Returns:
        List of tokenizer names
    """
    artifacts_path = get_artifacts_path()
    tokenizers_path = artifacts_path / "tokenizers"

    if not tokenizers_path.exists():
        return []

    return [d.name for d in tokenizers_path.iterdir() if d.is_dir()]


if __name__ == "__main__":
    # When run as a script, analyze all available tokenizers
    import argparse

    parser = argparse.ArgumentParser(description="Analyze tokenizer statistics from vocab.json and merges.txt files")
    parser.add_argument(
        "-t", "--tokenizer", type=str, help="Name of specific tokenizer to analyze (e.g., 'TinyStoriesV2-GPT4-train')"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output (don't show longest token details)"
    )
    parser.add_argument("-l", "--list", action="store_true", help="List all available tokenizers and exit")

    args = parser.parse_args()

    available_tokenizers = list_available_tokenizers()

    if not available_tokenizers:
        print("No tokenizers found in artifacts/tokenizers/")
        exit(1)

    # Handle --list option
    if args.list:
        print("Available tokenizers:")
        for tokenizer in available_tokenizers:
            print(f"  - {tokenizer}")
        exit(0)

    # Handle specific tokenizer analysis
    if args.tokenizer:
        if args.tokenizer not in available_tokenizers:
            print(f"Tokenizer '{args.tokenizer}' not found.")
            print("Available tokenizers:")
            for tokenizer in available_tokenizers:
                print(f"  - {tokenizer}")
            exit(1)
        analyze_tokenizer(args.tokenizer, verbose=not args.quiet)
    else:
        # Analyze all tokenizers
        print(f"Found {len(available_tokenizers)} tokenizer(s): {', '.join(available_tokenizers)}")
        print()
        for tokenizer_name in available_tokenizers:
            analyze_tokenizer(tokenizer_name, verbose=not args.quiet)
