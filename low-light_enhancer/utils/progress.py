"""Progress reporting utility built on tqdm."""

from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class ProgressTracker:
    """Thin wrapper around tqdm so the rest of the codebase doesn't import it directly."""

    def __init__(self, iterable: Iterable[T], description: str = "", total: int = None) -> None:
        self._iterable = iterable
        self._description = description
        self._total = total

    def __iter__(self) -> Iterator[T]:
        return iter(tqdm(self._iterable, desc=self._description, total=self._total))
