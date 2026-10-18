"""Bounded process-local cache for JSON-only direct-selection metadata.

The byte budget counts the exact retained UTF-8 JSON and key bytes, not Python
object/allocator overhead. Values are serialized on insertion and freshly
decoded on access, preventing callers from mutating cached audit records.
No tensors, model outputs or training activations can be stored by this API.
"""

from __future__ import annotations

from collections import OrderedDict
import json


class DirectSamplingCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024, max_entries: int = 2048):
        if any(type(value) is not int or value < 0 for value in (max_bytes, max_entries)):
            raise ValueError("cache capacities must be nonnegative integers")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = self._stores = self._oversize_skips = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return a fresh operational snapshot; it is not a sampling audit."""
        return {"hits": self._hits, "misses": self._misses, "evictions": self._evictions,
                "stores": self._stores, "oversize_skips": self._oversize_skips,
                "entries": len(self._entries), "bytes": self._bytes,
                "max_bytes": self.max_bytes, "max_entries": self.max_entries}

    def get(self, key: str) -> dict | None:
        if not isinstance(key, str):
            raise TypeError("cache keys must be strings")
        payload = self._entries.get(key)
        if payload is None:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return json.loads(payload)

    def put(self, key: str, payload: dict) -> bool:
        if not isinstance(key, str) or not isinstance(payload, dict):
            raise TypeError("cache entries require a string key and JSON object")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        size = len(key.encode("utf-8")) + len(encoded)
        if size > self.max_bytes or self.max_entries == 0:
            self._oversize_skips += 1
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= len(key.encode("utf-8")) + len(previous)
        while self._entries and (self._bytes + size > self.max_bytes
                                 or len(self._entries) >= self.max_entries):
            old_key, old_payload = self._entries.popitem(last=False)
            self._bytes -= len(old_key.encode("utf-8")) + len(old_payload)
            self._evictions += 1
        self._entries[key] = encoded
        self._bytes += size
        self._stores += 1
        return True

    def clear(self) -> None:
        """Release all cached values while preserving cumulative statistics."""
        self._entries.clear()
        self._bytes = 0
