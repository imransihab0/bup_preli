"""A single wall-clock budget for one request.

The judge treats a response beyond 30 s as a failure, so every optional step -
LLM retries, model escalation - must ask whether there is still time before it
spends any. This makes the worst case bounded instead of additive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Deadline:
    """Remaining time for the current request, in seconds."""

    budget: float
    started: float

    @classmethod
    def start(cls, budget: float) -> "Deadline":
        return cls(budget=budget, started=time.monotonic())

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.elapsed)

    def expired(self) -> bool:
        return self.remaining <= 0.0

    def allows(self, needed: float) -> bool:
        """True when `needed` seconds of work can still finish in budget."""
        return self.remaining >= needed

    def timeout_for(self, preferred: float, reserve: float = 0.0) -> float:
        """Clamp a call timeout to what the budget can actually afford.

        `reserve` holds time back for work that must still happen afterwards
        (the optimizer, serialization) so the response goes out in time.
        """
        return max(0.0, min(preferred, self.remaining - reserve))
