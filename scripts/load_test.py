#!/usr/bin/env python3
"""Concurrency check against a running service.

The judge may fire hidden cases in parallel; this confirms p95 holds up and
that nothing serializes behind a shared client.

    python scripts/load_test.py --concurrency 10 --requests 30
    python scripts/load_test.py --base https://your.onrender.com -c 5 -n 15
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "data" / "public_sample_cases.json").read_text())["cases"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    parser.add_argument("-n", "--requests", type=int, default=30)
    parser.add_argument("--unique", action="store_true",
                        help="Vary scenario_id so the interpretation cache never hits.")
    args = parser.parse_args()

    payloads = []
    for i in range(args.requests):
        case = json.loads(json.dumps(CASES[i % len(CASES)]["input"]))
        if args.unique:
            case["scenario_id"] = f"{case['scenario_id']}-{i}"
            # Perturb a note so the cache key differs too.
            case["operator_notes"][0] = case["operator_notes"][0] + f" (run {i})"
        payloads.append(case)

    results: list[tuple[float, int, str]] = []
    client = httpx.Client(timeout=40.0)

    def fire(payload):
        start = time.perf_counter()
        try:
            r = client.post(f"{args.base}/optimize-energy", json=payload)
            return time.perf_counter() - start, r.status_code, ""
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return time.perf_counter() - start, 0, type(exc).__name__

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(fire, payloads))
    wall = time.perf_counter() - wall

    latencies = sorted(r[0] for r in results)
    ok = sum(1 for r in results if r[1] == 200)
    failures = [r for r in results if r[1] != 200]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    print(f"base         {args.base}")
    print(f"concurrency  {args.concurrency}   requests {args.requests}"
          f"   cache-busting {'on' if args.unique else 'off'}")
    print(f"success      {ok}/{len(results)}")
    print(f"wall clock   {wall:.2f}s   throughput {len(results)/wall:.1f} req/s")
    print(f"latency      min {latencies[0]:.2f}s  median {statistics.median(latencies):.2f}s"
          f"  p95 {p95:.2f}s  max {latencies[-1]:.2f}s")
    if failures:
        print("failures:")
        for _, code, err in failures[:10]:
            print(f"  status={code} {err}")

    band = "3/3 (<=5s)" if p95 <= 5 else "2/3 (5-15s)" if p95 <= 15 else "1/3 or worse"
    print(f"judge latency band: {band}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
