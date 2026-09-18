"""API contract and robustness checks (S06, S10, and the Guide's Robustness row)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """These tests assert the HTTP contract, not model behaviour, so they run
    on the deterministic path. Scoped to this module - mutating the process
    environment here would silently disable the model for every other test."""
    monkeypatch.setattr(llm, "SETTINGS", replace(llm.SETTINGS, disable_llm=True))

CASES = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json").read_text()
)
SAMPLE = CASES["cases"][0]["input"]

client = TestClient(app, raise_server_exceptions=False)

REQUIRED_TOP_LEVEL = {
    "scenario_id", "directive_interpretation", "hourly_plan",
    "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary",
}
REQUIRED_HOUR_FIELDS = {
    "hour", "grid_kwh", "solar_used_kwh", "battery_action",
    "battery_kwh", "battery_energy_after_kwh",
}
REQUIRED_INTERPRETATION_FIELDS = {
    "note_index", "applies", "directive_type", "structured_adjustment", "explanation",
}


class TestHealth:
    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestResponseShape:
    @pytest.fixture(scope="class")
    @classmethod
    def body(cls):
        response = client.post("/optimize-energy", json=SAMPLE)
        assert response.status_code == 200
        return response.json()

    def test_top_level_fields_present(self, body):
        assert REQUIRED_TOP_LEVEL <= set(body)

    def test_scenario_id_is_echoed(self, body):
        assert body["scenario_id"] == SAMPLE["scenario_id"]

    def test_plan_covers_every_hour_once(self, body):
        hours = [entry["hour"] for entry in body["hourly_plan"]]
        assert hours == list(range(24))

    def test_hour_entries_have_required_fields(self, body):
        for entry in body["hourly_plan"]:
            assert REQUIRED_HOUR_FIELDS <= set(entry)
            assert entry["battery_action"] in {"charge", "discharge", "idle"}
            assert entry["grid_kwh"] >= 0 and entry["solar_used_kwh"] >= 0
            assert entry["battery_kwh"] >= 0
            if entry["battery_action"] == "idle":
                assert entry["battery_kwh"] == 0

    def test_one_interpretation_per_note_in_order(self, body):
        entries = body["directive_interpretation"]
        assert len(entries) == len(SAMPLE["operator_notes"])
        assert [e["note_index"] for e in entries] == list(range(len(entries)))
        for entry in entries:
            assert REQUIRED_INTERPRETATION_FIELDS <= set(entry)

    def test_applies_semantics(self, body):
        for entry in body["directive_interpretation"]:
            if entry["directive_type"] == "no_op":
                assert entry["applies"] is False
                assert entry["structured_adjustment"] is None
            else:
                assert entry["applies"] is True
                assert entry["structured_adjustment"] is not None
                assert entry["structured_adjustment"]["hours"] == sorted(
                    set(entry["structured_adjustment"]["hours"])
                )

    def test_totals_match_a_recalculation(self, body):
        tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in SAMPLE["hours"]}
        grid = sum(e["grid_kwh"] for e in body["hourly_plan"])
        cost = sum(e["grid_kwh"] * tariff[e["hour"]] for e in body["hourly_plan"])
        peak = max(e["grid_kwh"] for e in body["hourly_plan"])
        assert body["total_grid_kwh"] == pytest.approx(grid, abs=0.01)
        assert body["total_cost_bdt"] == pytest.approx(cost, abs=0.01)
        assert body["peak_grid_kwh"] == pytest.approx(peak, abs=0.01)


class TestRequestValidation:
    def test_malformed_json_is_400(self):
        response = client.post(
            "/optimize-energy",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "error" in response.json()

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.pop("battery"),
            lambda p: p.pop("scenario_id"),
            lambda p: p.update(operator_notes=[]),
            lambda p: p.update(operator_notes=["a", "b", "c", "d"]),
            lambda p: p.update(operator_notes=["  "]),
            lambda p: p.update(hours=p["hours"][:23]),
            lambda p: p["hours"][3].update(hour=25),
            lambda p: p["hours"][3].update(demand_kwh=-5),
        ],
        ids=[
            "missing-battery", "missing-scenario-id", "no-notes", "too-many-notes",
            "blank-note", "23-hours", "hour-out-of-range", "negative-demand",
        ],
    )
    def test_structurally_invalid_request_is_422(self, mutate):
        payload = json.loads(json.dumps(SAMPLE))
        mutate(payload)
        response = client.post("/optimize-energy", json=payload)
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_request"

    def test_error_responses_carry_a_request_id(self):
        response = client.post("/optimize-energy", json={"scenario_id": "X"})
        assert response.json()["request_id"]
        assert response.headers["x-request-id"]

    def test_supplied_request_id_is_echoed(self):
        response = client.post(
            "/optimize-energy", json=SAMPLE, headers={"x-request-id": "trace-me-123"}
        )
        assert response.headers["x-request-id"] == "trace-me-123"

    def test_error_responses_leak_nothing_sensitive(self):
        response = client.post("/optimize-energy", json={"scenario_id": "X"})
        text = response.text.lower()
        assert "traceback" not in text
        assert "openai" not in text
        assert "api_key" not in text and "sk-proj" not in text


class TestRobustness:
    def test_repeated_requests_stay_stable(self):
        for _ in range(5):
            assert client.post("/optimize-energy", json=SAMPLE).status_code == 200

    def test_unrecognized_extra_fields_are_ignored(self):
        payload = {**json.loads(json.dumps(SAMPLE)), "unexpected": {"deeply": ["nested"]}}
        assert client.post("/optimize-energy", json=payload).status_code == 200

    def test_every_public_case_returns_200(self):
        for case in CASES["cases"]:
            response = client.post("/optimize-energy", json=case["input"])
            assert response.status_code == 200, case["id"]
