"""Deterministic last-resort note parser.

SCOPE: this is a reliability safety net for the case where the language model is
unreachable mid-evaluation, so the service degrades instead of returning 5xx.
It is NOT the interpretation path - `llm.NoteInterpreter` is, and the Participant
Guide is explicit that phrase matching alone does not satisfy the requirement.
Keep it narrow; do not grow it into a competing interpreter.
"""

from __future__ import annotations

import re
from typing import Any

WORD_HOURS = {
    "midnight": 0, "noon": 12,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

FRACTION_WORDS = {
    "one-half": 0.5, "a half": 0.5, "half": 0.5,
    "one-third": 1 / 3, "a third": 1 / 3,
    "one-quarter": 0.25, "a quarter": 0.25,
    "one-fifth": 0.2, "a fifth": 0.2,
}

# Every group inside _TIME is non-capturing so the range regex below can rely on
# group(1) / group(2) being the two raw time strings.
_MERIDIEM = r"(?:a\.?m\.?|p\.?m\.?|in\s+the\s+(?:morning|afternoon|evening|night))"
_TIME = (
    rf"(?:(?<!\d)\d{{1,2}}(?::\d{{2}})?(?!\d)\s*{_MERIDIEM}?"
    rf"|{'|'.join(WORD_HOURS)})"
)
# A hyphenated range ("1-3 PM") carries no spaces; worded connectors need them,
# otherwise "to" would match inside words like "tomorrow".
_CONNECTOR = r"(?:\s*[-–—]+\s*|\s+(?:to|until|till|through|and)\s+)"
_RANGE = re.compile(
    rf"(?:from\s+|between\s+|starting\s+)?({_TIME}){_CONNECTOR}({_TIME})",
    re.IGNORECASE,
)

_REDUCTION = re.compile(
    r"reduction|reduce[sd]?\s+by|drop(?:s|ped|ping)?\s+by|down\s+by|"
    r"lower(?:ed)?\s+by|cut\s+by|loss\s+of",
    re.IGNORECASE,
)


def _meridiem_of(token: str) -> str | None:
    """Normalise both "6 PM" and "6 in the evening" to "am"/"pm"."""
    match = re.search(_MERIDIEM, token, re.IGNORECASE)
    if not match:
        return None
    found = match.group(0).replace(".", "").lower()
    if "morning" in found:
        return "am"
    if "afternoon" in found or "evening" in found or "night" in found:
        return "pm"
    return found.strip()


def _to_hour(token: str, fallback_meridiem: str | None) -> int | None:
    token = token.strip().lower()
    if token in WORD_HOURS:
        hour = WORD_HOURS[token]
        if token in ("noon", "midnight"):
            return hour
        meridiem = fallback_meridiem
    else:
        match = re.match(r"^(\d{1,2})(?::(\d{2}))?", token)
        if not match:
            return None
        hour = int(match.group(1))
        if hour > 23:
            return None
        meridiem = _meridiem_of(token) or fallback_meridiem
        # A bare hour above 12 is already 24-hour clock.
        if hour > 12:
            return hour

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour % 24


def _window(text: str) -> list[int]:
    """Extract one start-inclusive / end-exclusive whole-hour window."""
    match = _RANGE.search(text)
    if not match:
        return []

    start_raw, end_raw = match.group(1), match.group(2)
    end_meridiem = _meridiem_of(end_raw)
    start_meridiem = _meridiem_of(start_raw)

    # "6 until 9 PM" and "from one until three": an unqualified end of the
    # range borrows whichever half of the day the other end states, and a range
    # with no meridiem at all reads as campus afternoon.
    absolute = {"noon", "midnight"}
    start_absolute = start_raw.strip().lower() in absolute
    end_absolute = end_raw.strip().lower() in absolute

    shared = start_meridiem or end_meridiem
    end = _to_hour(end_raw, None if end_absolute else (end_meridiem or shared or "pm"))
    start = _to_hour(start_raw, None if start_absolute else (start_meridiem or shared or "pm"))
    if start is None or end is None:
        return []

    if start == end:
        return [start]

    span = (end - start) % 24 or 24
    # A borrowed meridiem can make a window wrap most of the day, which no
    # operator note means. Prefer whichever correction keeps the window short,
    # trying the start first - "hours 10 through 12" is a morning window, not a
    # late-evening one.
    if span > 12 and start_meridiem is None and not start_absolute and start >= 12:
        candidate = start - 12
        if ((end - candidate) % 24 or 24) <= 12:
            start, span = candidate, (end - candidate) % 24 or 24
    if span > 12 and end_meridiem is None and not end_absolute:
        candidate = (end + 12) % 24
        if candidate != start and ((candidate - start) % 24 or 24) <= 12:
            end, span = candidate, (candidate - start) % 24 or 24
    return sorted({(start + offset) % 24 for offset in range(span)})


def _factor(text: str) -> float | None:
    """Return the usable fraction that REMAINS, per S05.1."""
    value = _stated_fraction(text)
    if value is None:
        lowered = text.lower()
        if re.search(r"\b(no|zero)\s+(?:usable\s+)?(solar|output|generation|production)\b", lowered):
            return 0.0
        return None

    # "reduced by a third" states the LOSS, so the remainder is 1 - value. This
    # applies to worded fractions exactly as it does to percentages.
    if _REDUCTION.search(text):
        value = 1.0 - value
    return round(min(max(value, 0.0), 1.0), 6)


def _stated_fraction(text: str) -> float | None:
    percent = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text, re.IGNORECASE)
    if percent:
        return float(percent.group(1)) / 100.0
    lowered = text.lower()
    for phrase, value in FRACTION_WORDS.items():
        if phrase in lowered:
            return value
    return None


def _kwh_value(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def interpret(notes: list[str], battery: dict[str, Any]) -> list[dict[str, Any]]:
    return [_interpret_one(note, index, battery) for index, note in enumerate(notes)]


def _interpret_one(note: str, index: int, battery: dict[str, Any]) -> dict[str, Any]:
    text = note.strip()
    lowered = text.lower()
    hours = _window(text)
    base: dict[str, Any] = {"note_index": index, "hours": hours}

    mentions_solar = re.search(
        r"\b(solar|pv\b|panel|rooftop|photovoltaic|inverter|array|generation)", lowered
    )
    mentions_charge = re.search(r"\bcharg", lowered)
    mentions_discharge = re.search(r"\bdischarg", lowered)
    mentions_grid = re.search(r"\b(grid|import|intake|feeder|transformer|substation)", lowered)
    mentions_battery = re.search(r"\b(battery|storage|stored|reserve)", lowered)

    if not hours:
        return _no_op(index)

    if mentions_solar:
        factor = _factor(text)
        if factor is not None:
            return {**base, "directive_type": "solar_reduction", "factor": factor,
                    "explanation": f"Usable solar limited to {factor:.0%} of forecast."}

    if mentions_grid and re.search(
        r"\b(not exceed|no more than|at or below|cap|capped|limit|limited|max)", lowered
    ):
        cap = _kwh_value(text)
        if cap is not None:
            return {**base, "directive_type": "max_grid_window", "max_grid_kwh": cap,
                    "explanation": f"Grid import capped at {cap} kWh in the listed hours."}

    if mentions_battery and re.search(r"\b(at least|keep|maintain|remain|reserve|minimum)", lowered):
        percent = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text)
        if percent and re.search(r"capacit", lowered):
            minimum = float(percent.group(1)) / 100.0 * float(battery["capacity_kwh"])
        else:
            minimum = _kwh_value(text)
        if minimum is not None:
            return {**base, "directive_type": "minimum_battery_reserve",
                    "minimum_energy_kwh": minimum,
                    "explanation": f"Battery must hold at least {minimum:g} kWh in those hours."}

    negated = re.search(
        r"\b(not|no|avoid|disable[d]?|prohibit|isolat|unavailable|offline|"
        r"maintenance|inspect|testing|must not)", lowered
    )
    if mentions_discharge and negated:
        return {**base, "directive_type": "no_discharge_window",
                "explanation": "Battery discharging is unavailable in the listed hours."}
    if (mentions_charge or (mentions_battery and re.search(r"isolat|offline|out of service", lowered))) and negated:
        return {**base, "directive_type": "no_charge_window",
                "explanation": "Battery charging is unavailable in the listed hours."}

    return _no_op(index)


def _no_op(index: int) -> dict[str, Any]:
    return {
        "note_index": index,
        "directive_type": "no_op",
        "hours": [],
        "explanation": "This note does not affect today's 24-hour energy schedule.",
    }
