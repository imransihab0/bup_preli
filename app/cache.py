"""LRU cache for validated interpretations.

Judges replay scenarios - retries, reruns, tie-break checks - and each replay
otherwise pays for another model call. Caching removes that latency and cost
entirely for repeats.

What is cached is the POST-guardrail `Directive` tuple, never raw model output,
so a cache hit cannot bypass validation. Directives are frozen dataclasses, so
sharing an entry between requests is safe.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

from .directives import Directive


def interpretation_key(notes: list[str], battery: dict) -> str:
    """Stable key over everything that can change an interpretation.

    Battery capacity belongs in the key because relative language ("50% of
    capacity") resolves against it - the same note can mean different kWh on
    different scenarios.
    """
    payload = json.dumps(
        {"notes": [n.strip() for n in notes], "battery": battery},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InterpretationCache:
    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = max(0, maxsize)
        self._entries: OrderedDict[str, tuple[Directive, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[Directive] | None:
        if self._maxsize == 0:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return list(entry)

    def put(self, key: str, directives: list[Directive]) -> None:
        if self._maxsize == 0:
            return
        with self._lock:
            self._entries[key] = tuple(directives)
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    @property
    def size(self) -> int:
        return len(self._entries)
