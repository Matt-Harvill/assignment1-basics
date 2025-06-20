from collections import Counter
from sortedcontainers import SortedDict


class SortedBytePairFrequencyTracker:
    def __init__(self, bytes_pair_counts: Counter[tuple[bytes, bytes]]):
        self.counts: dict[tuple[bytes, bytes], int] = {}  # elem -> count
        self.sorted = SortedDict()  # count -> list of elems
        for byte_pair, count in bytes_pair_counts.items():
            self.add(byte_pair, count)

    def add(self, byte_pair: tuple[bytes, bytes], count: int):
        old = self.counts.get(byte_pair, 0)
        new = old + count
        self.counts[byte_pair] = new

        if old > 0:
            self.sorted[old].remove(byte_pair)
            if not self.sorted[old]:
                del self.sorted[old]

        if new not in self.sorted:
            self.sorted[new] = []
        self.sorted[new].append(byte_pair)

    def subtract(self, byte_pair: tuple[bytes, bytes], count: int):
        if byte_pair not in self.counts:
            return

        old = self.counts[byte_pair]

        self.sorted[old].remove(byte_pair)
        if not self.sorted[old]:
            del self.sorted[old]

        new = old - count
        self.counts[byte_pair] = new
        if new > 0:
            if new not in self.sorted:
                self.sorted[new] = []
            self.sorted[new].append(byte_pair)
        else:
            del self.counts[byte_pair]

    def get_most_frequent_byte_pair(self) -> tuple[bytes, bytes]:
        if not self.sorted:
            raise ValueError("No byte pairs in the tracker")
        return self.sorted.peekitem(-1)[1][-1]
