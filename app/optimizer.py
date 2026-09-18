"""Cost-minimal 24-hour schedule as a linear program.

Every GridWise rule in S09 is linear in the decision variables, so the whole
problem is an LP - no heuristics, no search. HiGHS returns the exact optimum
(verified against all ten organizer reference plans).

Decision variables, 96 in total, indexed hour-major within each block:
    grid[h]       grid energy purchased in hour h
    solar[h]      solar energy actually used in hour h
    charge[h]     energy added to the battery in hour h
    discharge[h]  energy removed from the battery in hour h
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .config import OUTPUT_PRECISION
from .directives import HOURS, CompiledConstraints
from .schemas import BatteryInput, HourInput

N = 24
GRID, SOLAR, CHARGE, DISCHARGE = 0, 1, 2, 3


class InfeasibleScheduleError(RuntimeError):
    """No schedule satisfies the supplied constraints."""


@dataclass
class Schedule:
    grid: list[float]
    solar_used: list[float]
    charge: list[float]
    discharge: list[float]
    energy_after: list[float]


def _index(block: int, hour: int) -> int:
    return block * N + hour


# Penalty weight for a unit of directive violation in the minimum-violation
# solve. It only has to dominate any cost saving a violation could buy; total
# daily cost is O(10^5), so this is several orders clear.
VIOLATION_PENALTY = 1e6


def solve(
    hours: list[HourInput],
    battery: BatteryInput,
    constraints: CompiledConstraints,
) -> Schedule:
    """Minimize SUM(grid[h] * tariff[h]) subject to every GridWise rule."""
    by_hour = {h.hour: h for h in hours}

    cost = np.zeros(4 * N)
    for hour in HOURS:
        cost[_index(GRID, hour)] = float(by_hour[hour].tariff_bdt_per_kwh)

    # --- Equalities ---------------------------------------------------------
    # Hourly energy balance (S09.5), rearranged so unknowns sit on the left:
    #   grid + solar_used + discharge - charge = demand
    a_eq = np.zeros((N + 1, 4 * N))
    b_eq = np.zeros(N + 1)
    for hour in HOURS:
        a_eq[hour, _index(GRID, hour)] = 1.0
        a_eq[hour, _index(SOLAR, hour)] = 1.0
        a_eq[hour, _index(DISCHARGE, hour)] = 1.0
        a_eq[hour, _index(CHARGE, hour)] = -1.0
        b_eq[hour] = float(by_hour[hour].demand_kwh)

    # End-of-day battery neutrality (S09.6): total charged == total discharged.
    for hour in HOURS:
        a_eq[N, _index(CHARGE, hour)] = 1.0
        a_eq[N, _index(DISCHARGE, hour)] = -1.0

    # --- Inequalities -------------------------------------------------------
    # Running state of charge after each hour must stay within
    # [min_energy_after[h], capacity]; the base minimum already sits in
    # min_energy_after, raised where a reserve directive applies (S09.2).
    a_ub = np.zeros((2 * N, 4 * N))
    b_ub = np.zeros(2 * N)
    running = np.zeros(4 * N)
    initial = float(battery.initial_energy_kwh)
    for hour in HOURS:
        running[_index(CHARGE, hour)] = 1.0
        running[_index(DISCHARGE, hour)] = -1.0
        a_ub[hour] = running
        b_ub[hour] = float(battery.capacity_kwh) - initial
        a_ub[N + hour] = -running
        b_ub[N + hour] = initial - constraints.min_energy_after[hour]

    # --- Bounds -------------------------------------------------------------
    # Grid caps, effective solar ceilings, and the no-charge/no-discharge
    # windows are all simple per-variable bounds (S05.3, S09.3, S09.4).
    bounds: list[tuple[float, float | None]] = []
    for hour in HOURS:
        cap = constraints.max_grid[hour]
        bounds.append((0.0, None if math.isinf(cap) else max(cap, 0.0)))
    for hour in HOURS:
        bounds.append((0.0, max(constraints.effective_solar[hour], 0.0)))
    for hour in HOURS:
        limit = float(battery.max_charge_kwh_per_hour) if constraints.charge_allowed[hour] else 0.0
        bounds.append((0.0, limit))
    for hour in HOURS:
        limit = (
            float(battery.max_discharge_kwh_per_hour)
            if constraints.discharge_allowed[hour]
            else 0.0
        )
        bounds.append((0.0, limit))

    result = linprog(
        cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not result.success:
        raise InfeasibleScheduleError(result.message or "linear program did not converge")

    return _materialize(result.x, hours, battery, constraints)


def solve_minimum_violation(
    hours: list[HourInput],
    battery: BatteryInput,
    constraints: CompiledConstraints,
) -> tuple[Schedule, float]:
    """Solve an over-constrained scenario by violating as little as possible.

    Organizer scoring scenarios are guaranteed feasible (S05.1), so this only
    runs when an interpretation is over-constrained or a scenario is genuinely
    impossible - for instance a grid cap below `demand - solar - max_discharge`,
    which no schedule can meet because the hourly discharge limit binds.

    Rather than discarding the directives wholesale, the numeric ones (grid cap
    and battery reserve) gain a heavily penalised slack variable. The solver
    then keeps every hour it CAN satisfy and exceeds only where physics forces
    it, by the smallest possible margin. Physical rules stay hard: energy
    balance, battery bounds, rate limits, effective solar, and the no-charge /
    no-discharge windows are never relaxed, because breaking those would trade
    a directive violation for a validity failure - and validity is checked on
    every case, not just the constrained ones.

    Returns the schedule and the total violation, in kWh.
    """
    by_hour = {h.hour: h for h in hours}
    capped = [h for h in HOURS if not math.isinf(constraints.max_grid[h])]
    reserved = [
        h for h in HOURS
        if constraints.min_energy_after[h] > float(battery.minimum_energy_kwh)
    ]
    # Layout: the usual 4N, then one slack per capped hour, then one per
    # reserved hour.
    n_slack = len(capped) + len(reserved)
    width = 4 * N + n_slack
    cap_slack = {hour: 4 * N + i for i, hour in enumerate(capped)}
    res_slack = {hour: 4 * N + len(capped) + i for i, hour in enumerate(reserved)}

    cost = np.zeros(width)
    for hour in HOURS:
        cost[_index(GRID, hour)] = float(by_hour[hour].tariff_bdt_per_kwh)
    for column in list(cap_slack.values()) + list(res_slack.values()):
        cost[column] = VIOLATION_PENALTY

    a_eq = np.zeros((N + 1, width))
    b_eq = np.zeros(N + 1)
    for hour in HOURS:
        a_eq[hour, _index(GRID, hour)] = 1.0
        a_eq[hour, _index(SOLAR, hour)] = 1.0
        a_eq[hour, _index(DISCHARGE, hour)] = 1.0
        a_eq[hour, _index(CHARGE, hour)] = -1.0
        b_eq[hour] = float(by_hour[hour].demand_kwh)
    for hour in HOURS:
        a_eq[N, _index(CHARGE, hour)] = 1.0
        a_eq[N, _index(DISCHARGE, hour)] = -1.0

    rows: list[np.ndarray] = []
    limits: list[float] = []
    running = np.zeros(width)
    initial = float(battery.initial_energy_kwh)
    for hour in HOURS:
        running[_index(CHARGE, hour)] = 1.0
        running[_index(DISCHARGE, hour)] = -1.0

        rows.append(running.copy())
        limits.append(float(battery.capacity_kwh) - initial)

        # Base minimum stays hard; a directive reserve above it may slip.
        floor = running.copy() * -1.0
        if hour in res_slack:
            floor[res_slack[hour]] = -1.0
        rows.append(floor)
        limits.append(initial - constraints.min_energy_after[hour])

        if hour in cap_slack:
            cap_row = np.zeros(width)
            cap_row[_index(GRID, hour)] = 1.0
            cap_row[cap_slack[hour]] = -1.0
            rows.append(cap_row)
            limits.append(max(constraints.max_grid[hour], 0.0))

    bounds: list[tuple[float, float | None]] = []
    bounds.extend((0.0, None) for _ in HOURS)                       # grid
    bounds.extend((0.0, max(constraints.effective_solar[h], 0.0)) for h in HOURS)
    for hour in HOURS:
        bounds.append(
            (0.0, float(battery.max_charge_kwh_per_hour) if constraints.charge_allowed[hour] else 0.0)
        )
    for hour in HOURS:
        bounds.append(
            (0.0, float(battery.max_discharge_kwh_per_hour)
             if constraints.discharge_allowed[hour] else 0.0)
        )
    bounds.extend((0.0, None) for _ in range(n_slack))

    result = linprog(
        cost, A_ub=np.array(rows), b_ub=limits, A_eq=a_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise InfeasibleScheduleError(
            result.message or "infeasible even with directive slack"
        )

    violation = float(sum(result.x[c] for c in list(cap_slack.values()) + list(res_slack.values())))
    return _materialize(result.x, hours, battery, constraints), violation


def _materialize(
    x: np.ndarray,
    hours: list[HourInput],
    battery: BatteryInput,
    constraints: CompiledConstraints,
) -> Schedule:
    """Turn the raw LP solution into a schedule that survives exact replay.

    Three corrections happen here, all of which matter for judging:

    1. Net out simultaneous charge and discharge. Degenerate optima can return
       both non-zero in the same hour, but `battery_action` allows only one.
    2. Clamp solar to the effective ceiling so floating-point noise cannot look
       like effective-solar overuse.
    3. Recompute grid from the balance equation rather than reporting the LP's
       own value, so the equality holds exactly at the emitted precision.
    """
    by_hour = {h.hour: h for h in hours}
    grid: list[float] = []
    solar_used: list[float] = []
    charge: list[float] = []
    discharge: list[float] = []
    energy_after: list[float] = []

    energy = float(battery.initial_energy_kwh)
    for hour in HOURS:
        net = x[_index(CHARGE, hour)] - x[_index(DISCHARGE, hour)]
        hour_charge = round(max(net, 0.0), OUTPUT_PRECISION)
        hour_discharge = round(max(-net, 0.0), OUTPUT_PRECISION)

        ceiling = max(constraints.effective_solar[hour], 0.0)
        hour_solar = round(min(max(x[_index(SOLAR, hour)], 0.0), ceiling), OUTPUT_PRECISION)
        hour_solar = min(hour_solar, ceiling)

        hour_grid = round(
            float(by_hour[hour].demand_kwh) + hour_charge - hour_solar - hour_discharge,
            OUTPUT_PRECISION,
        )
        hour_grid = max(hour_grid, 0.0)

        energy = round(energy + hour_charge - hour_discharge, OUTPUT_PRECISION)

        grid.append(hour_grid)
        solar_used.append(hour_solar)
        charge.append(hour_charge)
        discharge.append(hour_discharge)
        energy_after.append(energy)

    # Rounding can leave the final state a few 1e-7 off the initial level. Push
    # the residual into the last hour that has headroom so neutrality is exact.
    _restore_neutrality(
        grid, solar_used, charge, discharge, energy_after, hours, battery, constraints
    )
    return Schedule(grid, solar_used, charge, discharge, energy_after)


def _restore_neutrality(
    grid: list[float],
    solar_used: list[float],
    charge: list[float],
    discharge: list[float],
    energy_after: list[float],
    hours: list[HourInput],
    battery: BatteryInput,
    constraints: CompiledConstraints,
) -> None:
    initial = float(battery.initial_energy_kwh)
    residual = round(energy_after[-1] - initial, OUTPUT_PRECISION)
    if residual == 0.0:
        return

    by_hour = {h.hour: h for h in hours}
    for hour in reversed(list(HOURS)):
        # Shedding surplus means charging less; making up a deficit means
        # charging more. Either way only the charge leg needs to move.
        if residual > 0 and charge[hour] >= residual and constraints.charge_allowed[hour]:
            charge[hour] = round(charge[hour] - residual, OUTPUT_PRECISION)
        elif residual < 0 and discharge[hour] >= -residual and constraints.discharge_allowed[hour]:
            discharge[hour] = round(discharge[hour] + residual, OUTPUT_PRECISION)
        else:
            continue

        grid[hour] = max(
            round(
                float(by_hour[hour].demand_kwh) + charge[hour] - solar_used[hour] - discharge[hour],
                OUTPUT_PRECISION,
            ),
            0.0,
        )
        energy = initial
        for h in HOURS:
            energy = round(energy + charge[h] - discharge[h], OUTPUT_PRECISION)
            energy_after[h] = energy
        return
