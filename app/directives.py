"""Internal directive representation and its compilation into optimizer constraints.

This module is the single boundary between "what the LLM said" and "what the
optimizer solves". Nothing here trusts model output - everything arriving is
expected to have already passed `guardrails.validate_interpretations`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schemas import BatteryInput, DirectiveType, HourInput, StructuredAdjustment

HOURS = range(24)


@dataclass(frozen=True)
class Directive:
    """One validated operator directive, already normalized."""

    note_index: int
    directive_type: DirectiveType
    hours: tuple[int, ...] = ()
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None
    explanation: str = ""

    @property
    def applies(self) -> bool:
        return self.directive_type != "no_op"

    def structured_adjustment(self) -> dict | None:
        """Exact machine-checkable shape the judge expects for this type (S04)."""
        if not self.applies:
            return None
        adjustment = StructuredAdjustment(
            hours=list(self.hours),
            factor=self.factor if self.directive_type == "solar_reduction" else None,
            minimum_energy_kwh=(
                self.minimum_energy_kwh
                if self.directive_type == "minimum_battery_reserve"
                else None
            ),
            max_grid_kwh=(
                self.max_grid_kwh if self.directive_type == "max_grid_window" else None
            ),
        )
        return adjustment.serializable()


@dataclass
class CompiledConstraints:
    """Per-hour constraint arrays the optimizer and the replay validator share."""

    effective_solar: list[float] = field(default_factory=lambda: [0.0] * 24)
    charge_allowed: list[bool] = field(default_factory=lambda: [True] * 24)
    discharge_allowed: list[bool] = field(default_factory=lambda: [True] * 24)
    min_energy_after: list[float] = field(default_factory=lambda: [0.0] * 24)
    max_grid: list[float] = field(default_factory=lambda: [math.inf] * 24)


def compile_constraints(
    directives: list[Directive],
    hours: list[HourInput],
    battery: BatteryInput,
) -> CompiledConstraints:
    """Fold validated directives into the deterministic effects listed in S05.3.

    Overlapping directives of the same type resolve to the tighter requirement,
    which is the only reading that satisfies both at once.
    """
    by_hour = {h.hour: h for h in hours}
    compiled = CompiledConstraints(
        effective_solar=[float(by_hour[h].solar_kwh) for h in HOURS],
        min_energy_after=[float(battery.minimum_energy_kwh)] * 24,
    )

    for directive in directives:
        if not directive.applies:
            continue

        if directive.directive_type == "solar_reduction":
            factor = float(directive.factor or 0.0)
            for hour in directive.hours:
                reduced = float(by_hour[hour].solar_kwh) * factor
                compiled.effective_solar[hour] = min(compiled.effective_solar[hour], reduced)

        elif directive.directive_type == "no_charge_window":
            for hour in directive.hours:
                compiled.charge_allowed[hour] = False

        elif directive.directive_type == "no_discharge_window":
            for hour in directive.hours:
                compiled.discharge_allowed[hour] = False

        elif directive.directive_type == "minimum_battery_reserve":
            reserve = float(directive.minimum_energy_kwh or 0.0)
            for hour in directive.hours:
                compiled.min_energy_after[hour] = max(compiled.min_energy_after[hour], reserve)

        elif directive.directive_type == "max_grid_window":
            cap = float(directive.max_grid_kwh or 0.0)
            for hour in directive.hours:
                compiled.max_grid[hour] = min(compiled.max_grid[hour], cap)

    return compiled
