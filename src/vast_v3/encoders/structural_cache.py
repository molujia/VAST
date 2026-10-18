"""Instance-local LRU for immutable integer/tuple structural metadata only.

The conservative retained-entry budget includes recursive keys and values,
entry wrappers, and the actual live OrderedDict allocation. Shared objects are
charged repeatedly. Empty-container/control-object bookkeeping is excluded;
this is not a bound on process RSS or allocator/transient memory.
"""

from __future__ import annotations

from collections import OrderedDict
import sys
from typing import TypeAlias


Immutable: TypeAlias = int | tuple["Immutable", ...]


def _immutable_size(value: Immutable) -> int:
    if type(value) is int:
        return sys.getsizeof(value)
    if type(value) is tuple:
        return sys.getsizeof(value) + sum(_immutable_size(item) for item in value)
    raise TypeError("structural cache keys and values must contain only int and tuple")


class StructuralCache:
    """Bounded pure-Python cache; no module, tensor, fragment, or RNG state."""

    def __init__(self, max_entries: int = 2048, max_bytes: int = 16 * 1024 * 1024) -> None:
        if any(type(value) is not int or value < 0 for value in (max_entries, max_bytes)):
            raise ValueError("cache capacities must be nonnegative integers")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries: OrderedDict[Immutable, tuple[Immutable, int]] = OrderedDict()
        self._payload_bytes = 0
        self._hits = self._misses = self._stores = self._evictions = self._oversize_skips = 0

    @property
    def retained_bytes(self) -> int:
        """Conservative retained-entry bytes; zero for an empty cache."""
        return self._payload_bytes + sys.getsizeof(self._entries) if self._entries else 0

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "stores": self._stores,
                "evictions": self._evictions, "oversize_skips": self._oversize_skips,
                "entries": len(self._entries), "bytes": self.retained_bytes,
                "max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def get(self, key: Immutable) -> Immutable | None:
        _immutable_size(key)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: Immutable, value: Immutable) -> bool:
        payload_bytes = _immutable_size(key) + _immutable_size(value)
        # Each stored entry is (value, accounted_size). Count that wrapper and
        # its integer too; the integer's size may grow at Python digit bounds.
        size = payload_bytes + sys.getsizeof((value, 0)) + sys.getsizeof(payload_bytes)
        size = payload_bytes + sys.getsizeof((value, size)) + sys.getsizeof(size)
        minimum_dict_bytes = sys.getsizeof(OrderedDict(((0, 0),)))
        if self.max_entries == 0 or size + minimum_dict_bytes > self.max_bytes:
            self._oversize_skips += 1
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._payload_bytes -= previous[1]
        self._entries[key] = (value, size)
        self._payload_bytes += size
        # Insertion may grow the dict's table. Inspect its actual allocation
        # after insertion, not an estimated constant per-entry table cost.
        while self._entries and (len(self._entries) > self.max_entries
                                 or self.retained_bytes > self.max_bytes):
            _, removed = self._entries.popitem(last=False)
            self._payload_bytes -= removed[1]
            self._evictions += 1
        retained = key in self._entries
        if retained:
            self._stores += 1
        if not self._entries:
            self.clear()
        return retained

    def clear(self) -> None:
        """Release entries and any grown table; preserve cumulative counters."""
        self._entries.clear()
        self._payload_bytes = 0
