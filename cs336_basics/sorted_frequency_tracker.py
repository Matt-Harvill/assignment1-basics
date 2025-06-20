from sortedcontainers import SortedDict


class SortedBytePairFrequencyTracker:
    def __init__(self, bytes_pair_counts: dict[tuple[bytes, bytes], int]):
        self.sorted = SortedDict()  # count -> list of elems
        # Populate the sorted dict without modifying bytes_pair_counts
        for bytes_pair, count in bytes_pair_counts.items():
            if count not in self.sorted:
                self.sorted[count] = []
            self.sorted[count].append(bytes_pair)

    def add(self, bytes_pair: tuple[bytes, bytes], count: int, bytes_pair_counts: dict[tuple[bytes, bytes], int]):
        old = bytes_pair_counts.get(bytes_pair, 0)
        new = old + count
        bytes_pair_counts[bytes_pair] = new

        if old > 0:
            self.sorted[old].remove(bytes_pair)
            if not self.sorted[old]:
                del self.sorted[old]

        if new not in self.sorted:
            self.sorted[new] = []
        self.sorted[new].append(bytes_pair)

    def subtract(self, bytes_pair: tuple[bytes, bytes], count: int, bytes_pair_counts: dict[tuple[bytes, bytes], int]):
        if bytes_pair not in bytes_pair_counts:
            return

        old = bytes_pair_counts[bytes_pair]

        self.sorted[old].remove(bytes_pair)
        if not self.sorted[old]:
            del self.sorted[old]

        new = old - count
        bytes_pair_counts[bytes_pair] = new
        if new > 0:
            if new not in self.sorted:
                self.sorted[new] = []
            self.sorted[new].append(bytes_pair)
        else:
            del bytes_pair_counts[bytes_pair]

    def get_most_frequent_bytes_pair(self) -> tuple[bytes, bytes]:
        if not self.sorted:
            raise ValueError("No byte pairs in the tracker")

        # Most frequent byte pair is in the last item
        # We want to return the largest bytes pair (lexicographically highest)
        return max(self.sorted.peekitem()[1], key=lambda x: (x[0], x[1]))
