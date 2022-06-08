import queue
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Iterable, Sequence


@dataclass(frozen=True, eq=True)
class NeighborGroup:
    """Key for a group of potentially neighboring words.

    All words in the group have the same length, and the same letter at the same
    index.
    """

    length: int
    index: int
    letter: str


class Language:
    """A list of all valid words."""

    def __init__(
        self,
        word_list: Iterable[str],
        legal_letters: str = string.ascii_lowercase,
    ) -> None:
        self.table: DefaultDict[NeighborGroup, set[str]] = defaultdict(set)

        legal_letters = set(legal_letters)

        for word in word_list:
            if all(letter in legal_letters for letter in word):
                for (index, letter) in enumerate(word):
                    group = NeighborGroup(len(word), index, letter)
                    self.table[group].add(word)

    def find_neighbors(self, word: str) -> Iterable[str]:
        """Find all valid words that can be made by exchanging one letter."""
        for excluded_index in range(len(word)):
            neighbors = None
            for (index, letter) in enumerate(word):
                if index == excluded_index:
                    continue

                group = NeighborGroup(len(word), index, letter)
                entry = self.table[group]
                if neighbors is None:
                    neighbors = entry.copy()
                else:
                    neighbors.intersection_update(entry)

            neighbors.discard(word)
            yield from neighbors


@dataclass
class _ChangelingProblem:
    language: Language
    start: str
    target: str

    def __post_init__(self) -> None:
        if len(self.start) != len(self.target):
            raise ValueError

        self.marked: set[str] = set()
        self.queue = queue.PriorityQueue()

    def enqueue(self, path: Sequence[str]) -> None:
        end = path[-1]
        if end in self.marked:
            return
        self.marked.add(end)

        minimum_remaining_distance = sum(
            int(end[i] != self.target[i]) for i in range(len(self.target))
        )
        priority = (len(path), minimum_remaining_distance)
        self.queue.put((priority, tuple(path)))

    def solve(self) -> Sequence[str] | None:
        self.enqueue([self.start])
        while self.queue:
            priority, path = self.queue.get()
            cursor = path[-1]
            for neighbor in self.language.find_neighbors(cursor):
                next_path = list(path)
                next_path.append(neighbor)
                if neighbor == self.target:
                    return next_path
                self.enqueue(next_path)
        return None


def parse_dictionary_file(filepath: str) -> Iterable[str]:
    with open(filepath) as f:
        for line in f:
            word = line.strip()
            if word:
                yield word


def solve(language: Language, start: str, target: str) -> Sequence[str] | None:
    """Solve one instance of the "changelings" problem.

    In the given language, find the shortest sequence of valid words from
    `start` to `target` by exchanging one letter at a time. Return None if no
    solution exists.
    """
    return _ChangelingProblem(language, start, target).solve()
