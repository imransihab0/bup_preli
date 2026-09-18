"""Paraphrase robustness for the LLM interpretation path.

Skipped unless OPENAI_API_KEY is set, because it makes real model calls.
Run it before submitting:  pytest tests/test_paraphrases.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import SETTINGS

# SETTINGS reflects both the process environment and a local .env file.
pytestmark = pytest.mark.skipif(
    not SETTINGS.llm_enabled,
    reason="requires OPENAI_API_KEY (env or .env); exercises the live model",
)

SUITE = json.loads((Path(__file__).resolve().parent / "paraphrases.json").read_text())
TOLERANCE = 0.01


def _ids() -> list[str]:
    return [case["note"][:52] for case in SUITE["cases"]]


@pytest.mark.parametrize("case", SUITE["cases"], ids=_ids())
def test_note_resolves_to_expected_directive(case):
    from app.guardrails import validate_interpretations
    from app.llm import INTERPRETER
    from app.schemas import BatteryInput

    battery = BatteryInput.model_validate(SUITE["battery"])
    raw = INTERPRETER.interpret([case["note"]], battery)
    directive = validate_interpretations(raw, 1, battery)[0]
    expected = case["expect"]

    assert directive.directive_type == expected["directive_type"]
    if expected["directive_type"] == "no_op":
        assert directive.applies is False
        return

    assert directive.applies is True
    assert list(directive.hours) == expected["hours"]
    for key, attribute in (
        ("factor", "factor"),
        ("minimum_energy_kwh", "minimum_energy_kwh"),
        ("max_grid_kwh", "max_grid_kwh"),
    ):
        if key in expected:
            assert getattr(directive, attribute) == pytest.approx(expected[key], abs=TOLERANCE)
