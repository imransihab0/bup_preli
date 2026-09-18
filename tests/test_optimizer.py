"""Every directive type must bind in the returned schedule, not just the reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.directives import Directive, compile_constraints
from app.optimizer import InfeasibleScheduleError, solve
from app.replay import replay
from app.schemas import OptimizeRequest

CASES = json.loads((Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json").read_text())
BASE_INPUT = CASES["cases"][0]["input"]


def _request(**overrides) -> OptimizeRequest:
    return OptimizeRequest.model_validate({**BASE_INPUT, **overrides})


def _run(directives: list[Directive]):
    request = _request()
    hours = request.hours_sorted()
    constraints = compile_constraints(directives, hours, request.battery)
    schedule = solve(hours, request.battery, constraints)
    return request, hours, constraints, schedule


def _as_plan(schedule):
    from app.service import _to_plan

    return _to_plan(schedule)


class TestBaseConstraints:
    def test_unconstrained_plan_is_valid(self):
        request, hours, constraints, schedule = _run([])
        verdict = replay(_as_plan(schedule), hours, request.battery, constraints)
        assert verdict.ok, verdict.errors

    def test_battery_returns_to_initial_energy(self):
        request, _, _, schedule = _run([])
        assert schedule.energy_after[-1] == pytest.approx(
            request.battery.initial_energy_kwh, abs=0.01
        )

    def test_charge_and_discharge_never_share_an_hour(self):
        _, _, _, schedule = _run([])
        assert not any(c > 0 and d > 0 for c, d in zip(schedule.charge, schedule.discharge))


class TestDirectivesBind:
    def test_no_charge_window_holds(self):
        directive = Directive(0, "no_charge_window", hours=(2, 3, 4))
        _, _, _, schedule = _run([directive])
        assert all(schedule.charge[h] == 0 for h in (2, 3, 4))

    def test_no_discharge_window_holds(self):
        directive = Directive(0, "no_discharge_window", hours=(18, 19))
        _, _, _, schedule = _run([directive])
        assert all(schedule.discharge[h] == 0 for h in (18, 19))

    def test_max_grid_window_holds(self):
        # 175 is the tightest feasible cap for this scenario: hour 19 needs
        # 215 kWh with no solar and only 50 kWh of discharge available.
        directive = Directive(0, "max_grid_window", hours=(18, 19, 20), max_grid_kwh=175)
        _, _, _, schedule = _run([directive])
        assert all(schedule.grid[h] <= 175 + 0.01 for h in (18, 19, 20))

    def test_impossible_cap_is_reported_as_infeasible(self):
        # Hour 19 demands 215 kWh with zero solar and a 50 kWh discharge limit,
        # so a 100 kWh cap cannot be met. The solver must say so rather than
        # return a plan that quietly breaks the directive.
        directive = Directive(0, "max_grid_window", hours=(19,), max_grid_kwh=100)
        with pytest.raises(InfeasibleScheduleError):
            _run([directive])

    def test_minimum_reserve_holds(self):
        directive = Directive(0, "minimum_battery_reserve", hours=(18, 19, 20),
                              minimum_energy_kwh=150)
        _, _, _, schedule = _run([directive])
        assert all(schedule.energy_after[h] >= 150 - 0.01 for h in (18, 19, 20))

    def test_solar_reduction_caps_usable_solar(self):
        directive = Directive(0, "solar_reduction", hours=(12, 13), factor=0.25)
        request, hours, _, schedule = _run([directive])
        by_hour = {h.hour: h for h in hours}
        for hour in (12, 13):
            assert schedule.solar_used[hour] <= by_hour[hour].solar_kwh * 0.25 + 0.01

    def test_full_solar_blackout(self):
        directive = Directive(0, "solar_reduction", hours=(10, 11, 12), factor=0.0)
        _, _, _, schedule = _run([directive])
        assert all(schedule.solar_used[h] == 0 for h in (10, 11, 12))


class TestCombinations:
    def test_reserve_plus_grid_cap_both_hold(self):
        directives = [
            Directive(0, "minimum_battery_reserve", hours=(18, 19, 20, 21),
                      minimum_energy_kwh=90),
            Directive(1, "max_grid_window", hours=(19, 20), max_grid_kwh=180),
        ]
        request, hours, constraints, schedule = _run(directives)
        verdict = replay(_as_plan(schedule), hours, request.battery, constraints)
        assert verdict.ok, verdict.errors

    def test_overlapping_same_type_takes_the_tighter_bound(self):
        directives = [
            Directive(0, "max_grid_window", hours=(19,), max_grid_kwh=200),
            Directive(1, "max_grid_window", hours=(19,), max_grid_kwh=150),
        ]
        request = _request()
        constraints = compile_constraints(directives, request.hours_sorted(), request.battery)
        assert constraints.max_grid[19] == 150

    def test_overlapping_reserves_take_the_higher_floor(self):
        directives = [
            Directive(0, "minimum_battery_reserve", hours=(18,), minimum_energy_kwh=80),
            Directive(1, "minimum_battery_reserve", hours=(18,), minimum_energy_kwh=120),
        ]
        request = _request()
        constraints = compile_constraints(directives, request.hours_sorted(), request.battery)
        assert constraints.min_energy_after[18] == 120


class TestEdgeCases:
    def test_zero_solar_day(self):
        flat = [{**h, "solar_kwh": 0} for h in BASE_INPUT["hours"]]
        request = _request(hours=flat)
        constraints = compile_constraints([], request.hours_sorted(), request.battery)
        schedule = solve(request.hours_sorted(), request.battery, constraints)
        verdict = replay(_as_plan(schedule), request.hours_sorted(), request.battery, constraints)
        assert verdict.ok, verdict.errors

    def test_flat_tariff_still_yields_a_valid_plan(self):
        flat = [{**h, "tariff_bdt_per_kwh": 10} for h in BASE_INPUT["hours"]]
        request = _request(hours=flat)
        constraints = compile_constraints([], request.hours_sorted(), request.battery)
        schedule = solve(request.hours_sorted(), request.battery, constraints)
        verdict = replay(_as_plan(schedule), request.hours_sorted(), request.battery, constraints)
        assert verdict.ok, verdict.errors

    def test_battery_pinned_at_minimum_is_feasible(self):
        request = _request(battery={
            "capacity_kwh": 100, "initial_energy_kwh": 40, "minimum_energy_kwh": 40,
            "max_charge_kwh_per_hour": 20, "max_discharge_kwh_per_hour": 20,
        })
        constraints = compile_constraints([], request.hours_sorted(), request.battery)
        schedule = solve(request.hours_sorted(), request.battery, constraints)
        verdict = replay(_as_plan(schedule), request.hours_sorted(), request.battery, constraints)
        assert verdict.ok, verdict.errors
