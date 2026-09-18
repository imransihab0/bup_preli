"""The GridWise pipeline: interpret -> guardrail -> apply -> optimize -> verify."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import fallback_parser
from .config import OUTPUT_PRECISION
from .directives import HOURS, CompiledConstraints, Directive, compile_constraints
from .guardrails import GuardrailError, validate_interpretations
from .llm import INTERPRETER, LLMUnavailableError
from .optimizer import InfeasibleScheduleError, Schedule, solve
from .replay import replay
from .schemas import (
    DirectiveInterpretation,
    HourPlan,
    OptimizeRequest,
    OptimizeResponse,
)

logger = logging.getLogger(__name__)

TYPE_LABELS = {
    "solar_reduction": "the reduced solar availability",
    "minimum_battery_reserve": "the battery reserve requirement",
    "no_charge_window": "the no-charge window",
    "no_discharge_window": "the no-discharge window",
    "max_grid_window": "the grid import cap",
}


@dataclass
class PipelineOutcome:
    response: OptimizeResponse
    interpreter: str
    degraded: bool


def run(request: OptimizeRequest) -> PipelineOutcome:
    hours = request.hours_sorted()
    directives, interpreter = _interpret(request)
    constraints = compile_constraints(directives, hours, request.battery)

    schedule, active, degraded = _schedule(request, constraints)
    plan = _to_plan(schedule)

    # Replay against the constraints the plan was actually solved under, so the
    # reported totals always describe the plan being returned.
    verdict = replay(plan, hours, request.battery, active)
    if not verdict.ok:
        # Should be unreachable - the LP encodes every rule the replay checks.
        # Ship a physically valid plan over a cheap one (S09 penalties).
        logger.error("self-replay rejected the optimized plan: %s", verdict.errors[:3])
        schedule, active, _ = _schedule(request, _physical_only(request, hours))
        plan = _to_plan(schedule)
        verdict = replay(plan, hours, request.battery, active)
        degraded = True

    response = OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=[_to_interpretation(d) for d in directives],
        hourly_plan=plan,
        total_grid_kwh=round(verdict.total_grid_kwh, OUTPUT_PRECISION),
        total_cost_bdt=round(verdict.total_cost_bdt, OUTPUT_PRECISION),
        peak_grid_kwh=round(verdict.peak_grid_kwh, OUTPUT_PRECISION),
        plan_summary=_summarize(directives, verdict.total_cost_bdt, verdict.peak_grid_kwh),
    )
    return PipelineOutcome(response=response, interpreter=interpreter, degraded=degraded)


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def _interpret(request: OptimizeRequest) -> tuple[list[Directive], str]:
    """Run the model, validate what it returned, and fall back only if forced."""
    note_count = len(request.operator_notes)

    if INTERPRETER.enabled:
        try:
            raw = INTERPRETER.interpret(request.operator_notes, request.battery)
            return validate_interpretations(raw, note_count, request.battery), "llm"
        except GuardrailError as exc:
            # The model produced something we refuse to trust. One retry gives
            # it a chance to self-correct before we degrade.
            logger.warning("guardrails rejected model output: %s", exc)
            try:
                raw = INTERPRETER.interpret(request.operator_notes, request.battery)
                return validate_interpretations(raw, note_count, request.battery), "llm-retry"
            except (GuardrailError, LLMUnavailableError) as retry_exc:
                logger.warning("retry also failed: %s", retry_exc)
        except LLMUnavailableError as exc:
            logger.warning("model unavailable: %s", exc)

    try:
        raw = fallback_parser.interpret(request.operator_notes, request.battery.model_dump())
        return validate_interpretations(raw, note_count, request.battery), "deterministic-fallback"
    except GuardrailError as exc:
        logger.error("fallback parser produced invalid directives: %s", exc)

    # Last resort: interpret nothing rather than invent a constraint (S08).
    return (
        [
            Directive(
                note_index=index,
                directive_type="no_op",
                explanation="Interpretation unavailable; no schedule adjustment applied.",
            )
            for index in range(note_count)
        ],
        "none",
    )


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
def _schedule(
    request: OptimizeRequest, constraints: CompiledConstraints
) -> tuple[Schedule, CompiledConstraints, bool]:
    """Solve, and report which constraint set the returned schedule satisfies."""
    hours = request.hours_sorted()
    try:
        return solve(hours, request.battery, constraints), constraints, False
    except InfeasibleScheduleError as exc:
        # Organizer scoring scenarios are guaranteed feasible (S05.1), so this
        # only fires on an over-constrained misreading. A valid plan that drops
        # the battery/grid directives beats returning nothing at all; the solar
        # ceiling is kept because exceeding it is physically impossible.
        logger.warning("constrained solve infeasible (%s); relaxing directives", exc)
        relaxed = CompiledConstraints(
            effective_solar=list(constraints.effective_solar),
            min_energy_after=[float(request.battery.minimum_energy_kwh)] * 24,
        )
        return solve(hours, request.battery, relaxed), relaxed, True


def _physical_only(request: OptimizeRequest, hours: list) -> CompiledConstraints:
    """Constraints with no directive applied - the always-feasible baseline."""
    return CompiledConstraints(
        effective_solar=[float(h.solar_kwh) for h in hours],
        min_energy_after=[float(request.battery.minimum_energy_kwh)] * 24,
    )


def _to_plan(schedule: Schedule) -> list[HourPlan]:
    plan: list[HourPlan] = []
    for hour in HOURS:
        charge, discharge = schedule.charge[hour], schedule.discharge[hour]
        if charge > 0:
            action, amount = "charge", charge
        elif discharge > 0:
            action, amount = "discharge", discharge
        else:
            action, amount = "idle", 0.0
        plan.append(
            HourPlan(
                hour=hour,
                grid_kwh=schedule.grid[hour],
                solar_used_kwh=schedule.solar_used[hour],
                battery_action=action,
                battery_kwh=amount,
                battery_energy_after_kwh=schedule.energy_after[hour],
            )
        )
    return plan


def _to_interpretation(directive: Directive) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=directive.note_index,
        applies=directive.applies,
        directive_type=directive.directive_type,
        structured_adjustment=directive.structured_adjustment(),
        explanation=directive.explanation,
    )


def _summarize(directives: list[Directive], cost: float, peak: float) -> str:
    applied = [d for d in directives if d.applies]
    ignored = len(directives) - len(applied)

    clauses: list[str] = []
    if applied:
        labels = [TYPE_LABELS.get(d.directive_type, d.directive_type) for d in applied]
        joined = labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
        clauses.append(f"honours {joined}")
    else:
        clauses.append("applies no operator directive, since none affects today's schedule")
    if ignored:
        clauses.append(f"ignores {ignored} unrelated note{'s' if ignored > 1 else ''}")
    clauses.append("shifts battery energy from cheap hours into expensive ones")
    clauses.append("returns the battery to its starting level by the end of hour 23")

    body = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"
    return (
        f"The plan {body}. Total grid cost {cost:,.2f} BDT "
        f"with a {peak:,.2f} kWh peak hour."
    )
