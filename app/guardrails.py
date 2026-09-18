"""Deterministic validation of model output (Problem Statement S08).

LLM output is untrusted structured data. Every field is checked here before a
directive is allowed anywhere near the optimizer. A rejected interpretation
raises `GuardrailError` - it is never silently repaired into a new constraint.

Normalization (deduplicating and sorting hours, clamping -0.0 to 0.0) is applied
because it cannot introduce a constraint the model did not state. Anything that
would *change the meaning* of a directive is a rejection instead.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .directives import Directive
from .schemas import BatteryInput

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

HOUR_REQUIRED_TYPES = ALLOWED_TYPES - {"no_op"}


class GuardrailError(ValueError):
    """Raised when model output cannot be trusted as a directive."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise GuardrailError(f"{label} must be finite")
    return number


def _normalize_hours(raw: Any, label: str) -> tuple[int, ...]:
    """Hours must be unique integers 0-23 returned in ascending order (S08)."""
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise GuardrailError(f"{label}.hours must be a list of integers")

    hours: set[int] = set()
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise GuardrailError(f"{label}.hours must contain integers")
        if isinstance(item, float) and not item.is_integer():
            raise GuardrailError(f"{label}.hours must contain whole hours")
        hour = int(item)
        if not 0 <= hour <= 23:
            raise GuardrailError(f"{label}.hours entries must be within 0..23")
        hours.add(hour)

    if not hours:
        raise GuardrailError(f"{label}.hours must not be empty")
    return tuple(sorted(hours))


def _normalize_optional_hours(raw: Any, label: str) -> tuple[int, ...]:
    """Alternate-reading hours. Absent or empty is the normal case.

    Invalid content here is discarded rather than fatal: this field only widens
    optimizer constraints and never affects what gets reported, so a malformed
    value must not cost an otherwise-good interpretation.
    """
    if raw in (None, [], ()):
        return ()
    try:
        return _normalize_hours(raw, label)
    except GuardrailError:
        return ()


def validate_directive(raw: dict[str, Any], note_index: int, battery: BatteryInput) -> Directive:
    """Validate one raw model interpretation into a trusted `Directive`."""
    label = f"note {note_index}"

    directive_type = raw.get("directive_type")
    if directive_type not in ALLOWED_TYPES:
        raise GuardrailError(f"{label}: unsupported directive_type {directive_type!r}")

    explanation = raw.get("explanation") or ""
    if not isinstance(explanation, str):
        raise GuardrailError(f"{label}: explanation must be a string")

    # `applies` is derived, never taken from the model: no_op is the only type
    # permitted with applies = false, and every other type requires true (S05.1).
    if directive_type == "no_op":
        return Directive(
            note_index=note_index,
            directive_type="no_op",
            explanation=explanation.strip()
            or "This note does not affect today's 24-hour energy schedule.",
        )

    hours = _normalize_hours(raw.get("hours"), label)
    hedge_hours = _normalize_optional_hours(raw.get("alternate_hours"), label)
    factor = minimum_energy = max_grid = None

    if directive_type == "solar_reduction":
        factor = _finite_number(raw.get("factor"), f"{label}.factor")
        if not 0.0 <= factor <= 1.0:
            raise GuardrailError(f"{label}: factor must be between 0 and 1 inclusive")

    elif directive_type == "minimum_battery_reserve":
        minimum_energy = _finite_number(
            raw.get("minimum_energy_kwh"), f"{label}.minimum_energy_kwh"
        )
        if minimum_energy < 0:
            raise GuardrailError(f"{label}: minimum_energy_kwh must be non-negative")
        if minimum_energy > battery.capacity_kwh:
            raise GuardrailError(f"{label}: minimum_energy_kwh exceeds battery capacity")

    elif directive_type == "max_grid_window":
        max_grid = _finite_number(raw.get("max_grid_kwh"), f"{label}.max_grid_kwh")
        if max_grid < 0:
            raise GuardrailError(f"{label}: max_grid_kwh must be non-negative")

    return Directive(
        note_index=note_index,
        directive_type=directive_type,
        hours=hours,
        factor=factor,
        minimum_energy_kwh=minimum_energy,
        max_grid_kwh=max_grid,
        explanation=explanation.strip() or f"Interpreted as {directive_type}.",
        hedge_hours=hedge_hours,
    )


def validate_interpretations(
    raw_entries: list[dict[str, Any]],
    note_count: int,
    battery: BatteryInput,
) -> list[Directive]:
    """Validate the full interpretation set: one entry per note, in note order.

    Missing, duplicate, or out-of-range note_index values are interpretation
    failures (S08 "Note mapping") - not something to paper over.
    """
    if not isinstance(raw_entries, list):
        raise GuardrailError("interpretation payload must be a list")
    if len(raw_entries) != note_count:
        raise GuardrailError(
            f"expected exactly {note_count} interpretation entries, got {len(raw_entries)}"
        )

    by_index: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise GuardrailError(f"interpretation entry {position} is not an object")
        index = entry.get("note_index", position)
        if isinstance(index, bool) or not isinstance(index, (int, float)):
            raise GuardrailError(f"interpretation entry {position} has a non-numeric note_index")
        index = int(index)
        if not 0 <= index < note_count:
            raise GuardrailError(f"note_index {index} does not identify an operator note")
        if index in by_index:
            raise GuardrailError(f"note_index {index} appears more than once")
        by_index[index] = entry

    return [validate_directive(by_index[i], i, battery) for i in range(note_count)]
