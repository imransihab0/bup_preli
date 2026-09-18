"""Deadline budget, interpretation cache, hedging, and log-secret safety."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.cache import InterpretationCache, interpretation_key
from app.deadline import Deadline
from app.directives import Directive, compile_constraints
from app.logging_utils import RequestIdFilter, redact
from app.schemas import OptimizeRequest

CASES = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json").read_text()
)
SAMPLE = CASES["cases"][0]["input"]


class TestDeadline:
    def test_fresh_budget_allows_work(self):
        assert Deadline.start(20.0).allows(5.0)

    def test_expired_budget_blocks_work(self):
        spent = Deadline(budget=10.0, started=-1e9)
        assert spent.expired()
        assert not spent.allows(0.1)

    def test_timeout_is_clamped_to_remaining_budget(self):
        tight = Deadline(budget=3.0, started=Deadline.start(0).started)
        assert tight.timeout_for(preferred=12.0) == pytest.approx(3.0, abs=0.1)

    def test_reserve_is_held_back_for_post_call_work(self):
        d = Deadline(budget=10.0, started=Deadline.start(0).started)
        assert d.timeout_for(preferred=12.0, reserve=2.0) == pytest.approx(8.0, abs=0.1)

    def test_timeout_never_goes_negative(self):
        assert Deadline(budget=1.0, started=-1e9).timeout_for(12.0, reserve=5.0) == 0.0


class TestCache:
    def test_round_trip(self):
        cache = InterpretationCache(maxsize=4)
        directives = [Directive(0, "no_charge_window", hours=(2, 3))]
        cache.put("k", directives)
        assert cache.get("k")[0].hours == (2, 3)

    def test_miss_returns_none(self):
        assert InterpretationCache(maxsize=4).get("absent") is None

    def test_lru_evicts_oldest(self):
        cache = InterpretationCache(maxsize=2)
        for key in ("a", "b", "c"):
            cache.put(key, [])
        assert cache.get("a") is None and cache.size == 2

    def test_zero_size_disables_caching(self):
        cache = InterpretationCache(maxsize=0)
        cache.put("k", [])
        assert cache.get("k") is None

    def test_key_includes_battery_capacity(self):
        # "50% of capacity" resolves differently on a different pack.
        assert interpretation_key(["n"], {"capacity_kwh": 200}) != interpretation_key(
            ["n"], {"capacity_kwh": 400}
        )

    def test_key_ignores_surrounding_whitespace(self):
        assert interpretation_key(["  n  "], {"c": 1}) == interpretation_key(["n"], {"c": 1})

    def test_returned_list_is_a_copy(self):
        cache = InterpretationCache(maxsize=2)
        cache.put("k", [Directive(0, "no_op")])
        got = cache.get("k")
        got.append(Directive(1, "no_op"))
        assert len(cache.get("k")) == 1


class TestHedging:
    @pytest.fixture
    def request_obj(self):
        return OptimizeRequest.model_validate(SAMPLE)

    def test_reported_hours_exclude_the_hedge(self):
        d = Directive(0, "no_charge_window", hours=(14, 15), hedge_hours=(14, 15, 16))
        assert d.structured_adjustment() == {"hours": [14, 15]}

    def test_constraint_hours_take_the_union(self):
        d = Directive(0, "no_charge_window", hours=(14, 15), hedge_hours=(16,))
        assert d.constraint_hours(True) == (14, 15, 16)

    def test_hedging_can_be_disabled(self):
        d = Directive(0, "no_charge_window", hours=(14,), hedge_hours=(15,))
        assert d.constraint_hours(False) == (14,)

    def test_no_hedge_means_no_change(self):
        d = Directive(0, "no_charge_window", hours=(14, 15))
        assert d.constraint_hours(True) == (14, 15)

    @pytest.mark.parametrize(
        "directive,check",
        [
            (Directive(0, "no_charge_window", hours=(2,), hedge_hours=(3,)),
             lambda c: c.charge_allowed[3] is False),
            (Directive(0, "no_discharge_window", hours=(18,), hedge_hours=(19,)),
             lambda c: c.discharge_allowed[19] is False),
            (Directive(0, "max_grid_window", hours=(19,), hedge_hours=(20,), max_grid_kwh=150),
             lambda c: c.max_grid[20] == 150),
            (Directive(0, "minimum_battery_reserve", hours=(18,), hedge_hours=(19,),
                       minimum_energy_kwh=120),
             lambda c: c.min_energy_after[19] == 120),
            (Directive(0, "solar_reduction", hours=(12,), hedge_hours=(13,), factor=0.0),
             lambda c: c.effective_solar[13] == 0.0),
        ],
        ids=["no_charge", "no_discharge", "max_grid", "reserve", "solar"],
    )
    def test_hedge_widens_every_directive_type(self, request_obj, directive, check):
        constraints = compile_constraints(
            [directive], request_obj.hours_sorted(), request_obj.battery, hedging=True
        )
        assert check(constraints)

    def test_hedged_plan_is_valid_under_the_narrow_reading_too(self, request_obj):
        """The whole point: satisfying the union satisfies either reading."""
        from app.optimizer import solve
        from app.replay import replay
        from app.service import _to_plan

        directive = Directive(0, "no_charge_window", hours=(2, 3), hedge_hours=(4,))
        wide = compile_constraints(
            [directive], request_obj.hours_sorted(), request_obj.battery, hedging=True
        )
        schedule = solve(request_obj.hours_sorted(), request_obj.battery, wide)

        narrow = compile_constraints(
            [Directive(0, "no_charge_window", hours=(2, 3))],
            request_obj.hours_sorted(), request_obj.battery,
        )
        verdict = replay(_to_plan(schedule), request_obj.hours_sorted(),
                         request_obj.battery, narrow)
        assert verdict.ok, verdict.errors


class TestLogSecretSafety:
    @pytest.mark.parametrize(
        "text",
        [
            "key sk-proj-abcdefgh12345678 end",
            "sk-ant-api03-ZZZZZZZZZZZZ",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",
            'api_key="abcdefgh12345678"',
        ],
    )
    def test_credentials_are_redacted(self, text):
        assert "[redacted]" in redact(text)
        assert "abcdefgh12345678" not in redact(text)

    def test_ordinary_content_survives(self):
        message = "scenario=SAMPLE-01 hours=[12, 13] factor=0.25 cost=38365.00"
        assert redact(message) == message

    def test_filter_redacts_message_and_args(self):
        record = logging.LogRecord(
            "t", logging.INFO, __file__, 1,
            "using key %s", ("sk-proj-abcdefgh12345678",), None,
        )
        RequestIdFilter().filter(record)
        assert "abcdefgh12345678" not in record.getMessage()
        assert hasattr(record, "request_id")
