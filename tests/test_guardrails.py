"""Guardrails must reject untrusted model output rather than repair it (S08)."""

from __future__ import annotations

import pytest

from app.guardrails import GuardrailError, validate_directive, validate_interpretations
from app.schemas import BatteryInput

BATTERY = BatteryInput(
    capacity_kwh=200,
    initial_energy_kwh=100,
    minimum_energy_kwh=40,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)


def _directive(**overrides):
    base = {"directive_type": "no_charge_window", "hours": [2, 3], "explanation": "x"}
    return {**base, **overrides}


class TestNormalization:
    def test_hours_are_deduplicated_and_sorted(self):
        result = validate_directive(_directive(hours=[5, 2, 5, 3]), 0, BATTERY)
        assert result.hours == (2, 3, 5)

    def test_float_valued_whole_hours_accepted(self):
        assert validate_directive(_directive(hours=[2.0, 3.0]), 0, BATTERY).hours == (2, 3)

    def test_applies_is_derived_not_trusted(self):
        # A model claiming applies=false on a real directive must not win.
        result = validate_directive(_directive(applies=False), 0, BATTERY)
        assert result.applies is True

    def test_no_op_carries_null_adjustment(self):
        result = validate_directive({"directive_type": "no_op", "explanation": ""}, 0, BATTERY)
        assert result.applies is False
        assert result.structured_adjustment() is None


class TestRejection:
    @pytest.mark.parametrize(
        "payload",
        [
            _directive(directive_type="shed_load"),          # invented type
            _directive(hours=[]),                            # empty window
            _directive(hours=[24]),                          # out of range
            _directive(hours=[-1]),                          # out of range
            _directive(hours=[2.5]),                         # not a whole hour
            _directive(hours="2,3"),                         # wrong container
            {"directive_type": "solar_reduction", "hours": [1], "factor": 1.4},
            {"directive_type": "solar_reduction", "hours": [1], "factor": -0.1},
            {"directive_type": "solar_reduction", "hours": [1]},            # missing factor
            {"directive_type": "max_grid_window", "hours": [1], "max_grid_kwh": -5},
            {"directive_type": "minimum_battery_reserve", "hours": [1],
             "minimum_energy_kwh": 9_999},                                   # above capacity
            {"directive_type": "minimum_battery_reserve", "hours": [1],
             "minimum_energy_kwh": float("inf")},
        ],
    )
    def test_invalid_payload_raises(self, payload):
        with pytest.raises(GuardrailError):
            validate_directive(payload, 0, BATTERY)


class TestNoteMapping:
    def test_one_entry_per_note_in_order(self):
        entries = [
            {"note_index": 1, "directive_type": "no_op", "explanation": ""},
            {"note_index": 0, "directive_type": "no_charge_window", "hours": [1]},
        ]
        result = validate_interpretations(entries, 2, BATTERY)
        assert [d.note_index for d in result] == [0, 1]
        assert result[0].directive_type == "no_charge_window"

    def test_duplicate_note_index_rejected(self):
        entries = [{"note_index": 0, "directive_type": "no_op"}] * 2
        with pytest.raises(GuardrailError):
            validate_interpretations(entries, 2, BATTERY)

    def test_wrong_entry_count_rejected(self):
        with pytest.raises(GuardrailError):
            validate_interpretations([{"note_index": 0, "directive_type": "no_op"}], 2, BATTERY)

    def test_out_of_range_note_index_rejected(self):
        entries = [{"note_index": 7, "directive_type": "no_op"}]
        with pytest.raises(GuardrailError):
            validate_interpretations(entries, 1, BATTERY)
