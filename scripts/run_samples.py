#!/usr/bin/env python3
"""Run every public sample case through the full pipeline and grade the result.

    python scripts/run_samples.py                 # full pipeline (uses the LLM)
    python scripts/run_samples.py --no-llm        # deterministic fallback only
    python scripts/run_samples.py --case SAMPLE-03

Reports, per case: whether the interpretation matches organizer ground truth,
whether the returned plan survives replay against that ground truth, and how the
recalculated cost compares with the organizer optimum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CASES = ROOT / "data" / "public_sample_cases.json"
TOLERANCE = 0.01


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", action="append", dest="only", help="Run only this case id.")
    parser.add_argument("--no-llm", action="store_true", help="Force the deterministic fallback.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.no_llm:
        os.environ["GRIDWISE_DISABLE_LLM"] = "1"

    # Imported after the env override so config picks it up.
    from app.directives import compile_constraints
    from app.guardrails import validate_interpretations
    from app.replay import replay
    from app.schemas import OptimizeRequest
    from app.service import run

    pack = json.loads(args.cases.read_text())
    cases = [c for c in pack["cases"] if not args.only or c["id"] in args.only]

    passed = 0
    ratios: list[float] = []
    latencies: list[float] = []

    for case in cases:
        request = OptimizeRequest.model_validate(case["input"])
        expected = case["expected_output"]

        started = time.perf_counter()
        outcome = run(request)
        latency = time.perf_counter() - started
        latencies.append(latency)
        response = outcome.response

        problems: list[str] = []

        # 1. Interpretation vs organizer ground truth.
        problems.extend(
            _compare_interpretation(
                [d.model_dump() for d in response.directive_interpretation],
                expected["directive_interpretation"],
            )
        )

        # 2. Plan replayed against GROUND-TRUTH directives, not our own reading -
        #    this is exactly what the judge does.
        truth = validate_interpretations(
            [_flatten(entry) for entry in expected["directive_interpretation"]],
            len(request.operator_notes),
            request.battery,
        )
        constraints = compile_constraints(truth, request.hours_sorted(), request.battery)
        verdict = replay(response.hourly_plan, request.hours_sorted(), request.battery, constraints)
        problems.extend(verdict.errors)

        # 3. Reported totals must match a recalculation from hourly_plan.
        for field, actual in (
            ("total_grid_kwh", verdict.total_grid_kwh),
            ("total_cost_bdt", verdict.total_cost_bdt),
            ("peak_grid_kwh", verdict.peak_grid_kwh),
        ):
            if abs(getattr(response, field) - actual) > TOLERANCE:
                problems.append(f"{field} reported {getattr(response, field)}, recalculated {actual}")

        # 4. Cost quality against the organizer optimum.
        optimal = float(expected["total_cost_bdt"])
        ratio = 1.0 if not verdict.ok else min(1.0, optimal / max(verdict.total_cost_bdt, 1e-9))
        if verdict.ok:
            ratios.append(ratio)

        status = "PASS" if not problems else "FAIL"
        passed += not problems
        delta = verdict.total_cost_bdt - optimal
        print(
            f"{status}  {case['id']:<10} {case['label'][:34]:<34} "
            f"cost {verdict.total_cost_bdt:>9,.2f} vs {optimal:>9,.2f} "
            f"({delta:+.2f})  ratio {ratio:.4f}  {latency:5.2f}s  [{outcome.interpreter}]"
        )
        for problem in problems[:6]:
            print(f"        - {problem}")
        if args.verbose:
            for entry in response.directive_interpretation:
                print(f"        note {entry.note_index}: {entry.directive_type} "
                      f"{entry.structured_adjustment}")

    print()
    print(f"{passed}/{len(cases)} cases fully valid")
    if ratios:
        score = 10 * sum(ratios) / len(ratios)
        print(f"Optimization Quality (10 pts): {score:.2f}")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        print(f"latency  mean {sum(latencies)/len(latencies):.2f}s  p95 {p95:.2f}s")
    return 0 if passed == len(cases) else 1


def _flatten(entry: dict) -> dict:
    """Reference entries nest their numbers; guardrails take them flat."""
    adjustment = entry.get("structured_adjustment") or {}
    return {
        "note_index": entry["note_index"],
        "directive_type": entry["directive_type"],
        "hours": adjustment.get("hours", []),
        "factor": adjustment.get("factor"),
        "minimum_energy_kwh": adjustment.get("minimum_energy_kwh"),
        "max_grid_kwh": adjustment.get("max_grid_kwh"),
        "explanation": entry.get("explanation", ""),
    }


def _compare_interpretation(actual: list[dict], expected: list[dict]) -> list[str]:
    """Free-text explanation wording is not compared - only machine-checkable fields."""
    problems: list[str] = []
    if len(actual) != len(expected):
        return [f"expected {len(expected)} interpretation entries, got {len(actual)}"]

    for got, want in zip(actual, expected):
        index = want["note_index"]
        if got["note_index"] != index:
            problems.append(f"note {index}: entries out of order")
        if got["applies"] != want["applies"]:
            problems.append(f"note {index}: applies {got['applies']}, expected {want['applies']}")
        if got["directive_type"] != want["directive_type"]:
            problems.append(
                f"note {index}: directive_type {got['directive_type']}, "
                f"expected {want['directive_type']}"
            )
            continue

        got_adj, want_adj = got["structured_adjustment"], want["structured_adjustment"]
        if (got_adj is None) != (want_adj is None):
            problems.append(f"note {index}: structured_adjustment presence mismatch")
            continue
        if want_adj is None:
            continue
        if got_adj.get("hours") != want_adj.get("hours"):
            problems.append(
                f"note {index}: hours {got_adj.get('hours')}, expected {want_adj.get('hours')}"
            )
        for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key in want_adj:
                if key not in got_adj:
                    problems.append(f"note {index}: missing {key}")
                elif abs(float(got_adj[key]) - float(want_adj[key])) > TOLERANCE:
                    problems.append(
                        f"note {index}: {key} {got_adj[key]}, expected {want_adj[key]}"
                    )
            elif key in got_adj:
                problems.append(f"note {index}: unexpected {key} in structured_adjustment")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
