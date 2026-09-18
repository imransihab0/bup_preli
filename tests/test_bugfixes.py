"""Regressions for bugs found by external review.

Each test names the behaviour that was wrong, so a reintroduction is obvious.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.fallback_parser import _factor, _window
from app.guardrails import GuardrailError, validate_interpretations
from app.schemas import BatteryInput, OptimizeRequest

CASES = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json").read_text()
)
SAMPLE = CASES["cases"][0]["input"]
BATTERY = BatteryInput(
    capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40,
    max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
)


class TestMinimumViolationPlanIsActuallyShipped:
    """Bug 1: the min-violation plan was replayed against the strict directives
    it knowingly violates, so it always failed and the service shipped a
    physical-only plan that ignored the directive entirely."""

    def test_impossible_cap_ships_the_minimum_violation(self):
        from app.service import run

        payload = json.loads(json.dumps(SAMPLE))
        payload["scenario_id"] = "impossible-cap"
        payload["operator_notes"] = [
            "From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour."
        ]
        outcome = run(OptimizeRequest.model_validate(payload))
        plan = {entry.hour: entry for entry in outcome.response.hourly_plan}

        # hour 19 floor is demand 215 - solar 0 - discharge 50 = 165.
        assert plan[19].grid_kwh == pytest.approx(165.0, abs=0.01)
        # The hours that CAN meet the cap must still meet it.
        assert plan[18].grid_kwh <= 155.01
        assert plan[20].grid_kwh <= 155.01
        total_violation = sum(max(plan[h].grid_kwh - 155, 0) for h in (18, 19, 20))
        assert total_violation == pytest.approx(10.0, abs=0.01)
        assert outcome.degraded is True

    def test_degraded_summary_does_not_claim_the_directive_was_honoured(self):
        from app.service import run

        payload = json.loads(json.dumps(SAMPLE))
        payload["scenario_id"] = "impossible-cap-summary"
        payload["operator_notes"] = [
            "From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour."
        ]
        outcome = run(OptimizeRequest.model_validate(payload))
        assert "honours" not in outcome.response.plan_summary
        assert "physically allow" in outcome.response.plan_summary


class TestFallbackFactor:
    """Bug 2: worded fractions skipped the reduction test that percentages used,
    so "reduced by a third" returned the loss instead of the remainder."""

    @pytest.mark.parametrize(
        "note,expected",
        [
            ("solar reduced by a third", 2 / 3),
            ("output drops by a quarter", 0.75),
            ("cut by half for maintenance", 0.5),
            ("about half the forecast output", 0.5),
            ("expect an 80% reduction", 0.2),
            ("drops to 20% of forecast", 0.2),
            ("one-fifth of normal output", 0.2),
        ],
    )
    def test_remaining_fraction(self, note, expected):
        assert _factor(note) == pytest.approx(expected, abs=0.001)

    def test_no_float_noise_in_the_emitted_value(self):
        # 0.19999999999999996 would be reported verbatim in the response.
        assert repr(_factor("an 80% reduction")) == "0.2"


class TestFallbackWindow:
    """Bug 3: absolute times and verbal meridiems were mishandled, producing
    day-long windows or an empty window that became a false no_op."""

    @pytest.mark.parametrize(
        "note,expected",
        [
            ("from midnight until 2", [0, 1]),
            ("from 6 in the evening to 9 in the evening", [18, 19, 20]),
            ("from 9 in the morning to 11 in the morning", [9, 10]),
            ("from 1 PM to 3 PM", [13, 14]),
            ("noon until 2 PM", [12, 13]),
            ("10 PM until 2 AM", [0, 1, 22, 23]),
            ("2 AM until 5 AM", [2, 3, 4]),
        ],
    )
    def test_window(self, note, expected):
        assert _window(note) == expected

    def test_overnight_window_stays_short(self):
        # Previously produced a five-hour daytime window instead of wrapping.
        hours = _window("from 23 until 4 in the morning")
        assert hours == [0, 1, 2, 3, 23]

    def test_isolated_battery_is_a_no_charge_window(self):
        from app.fallback_parser import interpret

        result = interpret(
            ["The battery is isolated from 2 until 5 AM."], BATTERY.model_dump()
        )[0]
        assert result["directive_type"] == "no_charge_window"
        assert result["hours"] == [2, 3, 4]


class TestNoteIndexStrictness:
    """Bug 4: a fractional note_index was silently truncated, while fractional
    hours were correctly rejected."""

    def test_fractional_note_index_is_rejected(self):
        entries = [
            {"note_index": 1.5, "directive_type": "no_op"},
            {"note_index": 0, "directive_type": "no_op"},
        ]
        with pytest.raises(GuardrailError):
            validate_interpretations(entries, 2, BATTERY)

    def test_float_valued_whole_note_index_is_accepted(self):
        entries = [
            {"note_index": 1.0, "directive_type": "no_op"},
            {"note_index": 0.0, "directive_type": "no_op"},
        ]
        assert len(validate_interpretations(entries, 2, BATTERY)) == 2


class TestNumericHardening:
    """Bug 5: non-finite or negative tariffs reached the solver and surfaced as
    a 500 rather than a clean 422."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -5.0])
    def test_bad_tariff_is_rejected(self, bad):
        payload = json.loads(json.dumps(SAMPLE))
        payload["hours"][3]["tariff_bdt_per_kwh"] = bad
        with pytest.raises(ValidationError):
            OptimizeRequest.model_validate(payload)

    @pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_hour_values_are_rejected(self, field, bad):
        payload = json.loads(json.dumps(SAMPLE))
        payload["hours"][3][field] = bad
        with pytest.raises(ValidationError):
            OptimizeRequest.model_validate(payload)

    def test_zero_tariff_is_still_allowed(self):
        payload = json.loads(json.dumps(SAMPLE))
        payload["hours"][3]["tariff_bdt_per_kwh"] = 0.0
        assert OptimizeRequest.model_validate(payload)
