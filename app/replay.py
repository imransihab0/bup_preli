"""Judge-equivalent replay of a finished plan (Problem Statement S11.2-S11.3).

The service runs this against its own response before returning it. If the plan
cannot survive replay we fall back rather than ship an invalid schedule - an
invalid case scores zero for directive application *and* optimization, so a
slightly worse valid plan always beats a cheaper broken one.

`scripts/run_samples.py` reuses this to check the public cases offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import JUDGE_TOLERANCE
from .directives import HOURS, CompiledConstraints
from .schemas import BatteryInput, HourInput, HourPlan


@dataclass
class ReplayResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    total_grid_kwh: float = 0.0
    total_cost_bdt: float = 0.0
    peak_grid_kwh: float = 0.0


def replay(
    plan: list[HourPlan],
    hours: list[HourInput],
    battery: BatteryInput,
    constraints: CompiledConstraints,
    tolerance: float = JUDGE_TOLERANCE,
) -> ReplayResult:
    errors: list[str] = []
    by_hour = {h.hour: h for h in hours}
    plan_by_hour = {entry.hour: entry for entry in plan}

    if set(plan_by_hour) != set(HOURS) or len(plan) != 24:
        return ReplayResult(
            ok=False, errors=["hourly_plan must contain exactly 24 unique hours 0..23"]
        )

    energy = float(battery.initial_energy_kwh)
    total_grid = total_cost = peak = 0.0

    for hour in HOURS:
        entry = plan_by_hour[hour]
        source = by_hour[hour]

        if entry.grid_kwh < -tolerance or entry.solar_used_kwh < -tolerance:
            errors.append(f"hour {hour}: negative energy value")
        if entry.battery_kwh < -tolerance:
            errors.append(f"hour {hour}: negative battery_kwh")

        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
        if entry.battery_action == "idle" and abs(entry.battery_kwh) > tolerance:
            errors.append(f"hour {hour}: idle requires battery_kwh = 0")

        supply = entry.grid_kwh + entry.solar_used_kwh + discharge
        demand = float(source.demand_kwh) + charge
        if abs(supply - demand) > tolerance:
            errors.append(f"hour {hour}: energy balance off by {supply - demand:.4f} kWh")

        if entry.solar_used_kwh > constraints.effective_solar[hour] + tolerance:
            errors.append(
                f"hour {hour}: solar_used {entry.solar_used_kwh:.3f} exceeds effective solar "
                f"{constraints.effective_solar[hour]:.3f}"
            )

        if charge > float(battery.max_charge_kwh_per_hour) + tolerance:
            errors.append(f"hour {hour}: charge exceeds hourly limit")
        if discharge > float(battery.max_discharge_kwh_per_hour) + tolerance:
            errors.append(f"hour {hour}: discharge exceeds hourly limit")
        if charge > tolerance and not constraints.charge_allowed[hour]:
            errors.append(f"hour {hour}: charging during a no_charge_window")
        if discharge > tolerance and not constraints.discharge_allowed[hour]:
            errors.append(f"hour {hour}: discharging during a no_discharge_window")
        if entry.grid_kwh > constraints.max_grid[hour] + tolerance:
            errors.append(f"hour {hour}: grid import exceeds the max_grid_window cap")

        energy = energy + charge - discharge
        if abs(energy - entry.battery_energy_after_kwh) > tolerance:
            errors.append(f"hour {hour}: battery_energy_after_kwh disagrees with the transition")
        if energy < constraints.min_energy_after[hour] - tolerance:
            errors.append(
                f"hour {hour}: battery energy {energy:.3f} below the required minimum "
                f"{constraints.min_energy_after[hour]:.3f}"
            )
        if energy > float(battery.capacity_kwh) + tolerance:
            errors.append(f"hour {hour}: battery energy above capacity")

        total_grid += entry.grid_kwh
        total_cost += entry.grid_kwh * float(source.tariff_bdt_per_kwh)
        peak = max(peak, entry.grid_kwh)

    if abs(energy - float(battery.initial_energy_kwh)) > tolerance:
        errors.append(
            f"end-of-day battery energy {energy:.3f} does not return to "
            f"{float(battery.initial_energy_kwh):.3f}"
        )

    return ReplayResult(
        ok=not errors,
        errors=errors,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak,
    )
