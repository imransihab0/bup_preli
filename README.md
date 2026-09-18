# GridWise — LLM-Assisted Smart Campus Energy Optimization

**BUP CSE Fest 2026 Hackathon · Online Preliminary**

A deployed HTTP service that reads free-text campus operator notes, converts them
into machine-checkable directives with an LLM, validates those directives
deterministically, and returns a cost-minimal 24-hour energy schedule that obeys
them.

The optimizer is an exact linear program, not a heuristic: it reproduces the
organizer's reference cost on **all ten public cases to the cent**.

| | |
|---|---|
| **Health endpoint** | `GET /health` → `{"status":"ok"}` |
| **Main endpoint** | `POST /optimize-energy` |
| **Model** | `gpt-5.6-luna` (OpenAI Responses API, schema-constrained output) |
| **Optimizer** | Linear program, 96 variables, HiGHS via SciPy |
| **Docker image** | `imransihab0/gridwise-api:1.0.0` |
| **Tests** | 112 offline + 43 live interpretation checks |

---

## Table of contents

1. [Quickstart](#quickstart) · 2. [Verify it works](#verify-it-works) ·
3. [Architecture](#architecture) · 4. [The LLM's role](#the-llms-role) ·
5. [Guardrails](#guardrails) · 6. [Optimizer](#optimizer) ·
7. [Robustness](#robustness) · 8. [API contract](#api-contract) ·
9. [Configuration](#configuration) · 10. [Deployment](#deployment) ·
11. [Docker fallback](#docker-fallback) · 12. [Testing](#testing) ·
13. [Project layout](#project-layout) · 14. [Dependencies](#dependencies) ·
15. [Known limitations](#known-limitations)

---

## Quickstart

Requires **Python 3.11+** (3.12 recommended) and an **OpenAI API key**.

```bash
git clone https://github.com/imransihab0/bup_preli.git
cd bup_preli

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key on the OPENAI_API_KEY line
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`.env` is gitignored and loaded automatically at startup. A real environment
variable always wins over the file, so Render and Docker — which inject
`OPENAI_API_KEY` directly — need no `.env` at all.

Expected startup line:

```
GridWise ready | model=gpt-5.6-luna | escalation=gpt-5.6-terra | llm_enabled=True | budget=20s | hedging=True | cache=256
```

---

## Verify it works

In a second terminal:

```bash
# 1. readiness
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}

# 2. one public sample case, end to end
curl -s -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json;print(json.dumps(json.load(open('data/public_sample_cases.json'))['cases'][0]['input']))")"
```

**Expected:** HTTP 200, `scenario_id: "SAMPLE-01"`, a 24-entry `hourly_plan`,
`total_cost_bdt` of **38365.00**, and two `directive_interpretation` entries —
`solar_reduction` over hours `[12, 13]` with `factor 0.25`, and a `no_op`.

Or run everything at once, including the error-code cases:

```bash
./scripts/smoke_test.sh                            # local
./scripts/smoke_test.sh https://<your-service>     # deployed
```

---

## Architecture

```
 operator_notes (natural language)
        │
        ▼
 ┌──────────────────┐   The model interprets every note into a flat structured
 │  app/llm.py      │   directive, with a confidence flag. One call covers all
 │  app/prompts.py  │   1-3 notes. Output is UNTRUSTED from here on.
 └──────────────────┘
        │  └── low confidence + budget available → one retry on a stronger model
        ▼
 ┌──────────────────┐   Deterministic validation: allowed types only, one entry
 │ app/guardrails.py│   per note in order, hours unique ints 0-23 ascending,
 └──────────────────┘   factor in [0,1], reserve ≤ capacity, cap ≥ 0.
        │               `applies` is DERIVED here, never taken from the model.
        ▼
 ┌──────────────────┐   Validated directives are cached (post-guardrail, so a
 │  app/cache.py    │   hit can never bypass validation) and folded into
 │ app/directives.py│   per-hour constraint arrays: effective solar,
 └──────────────────┘   charge/discharge windows, energy floor, grid cap.
        │               Overlaps resolve to the tighter bound.
        ▼
 ┌──────────────────┐   96-variable linear program solved with HiGHS.
 │ app/optimizer.py │   Exact optimum — no heuristics, no search.
 └──────────────────┘   Over-constrained input falls back to a
        │               minimum-violation solve rather than dropping directives.
        ▼
 ┌──────────────────┐   The finished plan is replayed against every judge rule
 │  app/replay.py   │   BEFORE it is returned. A plan that fails replay is
 └──────────────────┘   never shipped.
        │
        ▼
     JSON response

 app/deadline.py     one wall-clock budget spanning the whole pipeline
 app/logging_utils.py per-request id on every log line, secrets redacted
 app/fallback_parser.py deterministic safety net if the model is unreachable
```

`app/service.py` wires these together; `app/main.py` is the FastAPI surface.

---

## The LLM's role

The model performs the operator-note interpretation itself — it decides whether
a note applies, which directive type it is, which hours it covers, and what the
numeric value is. That structured output is what the optimizer's constraints are
built from.

**The model is not used for `plan_summary` or any cosmetic text**; the summary is
generated deterministically from the directives that were actually applied. Using
an LLM only for prose would not satisfy the challenge requirement.

**Model:** `gpt-5.6-luna` via the official `openai` Python SDK, using
`client.responses.parse` with a Pydantic `text_format`, so the response is
schema-constrained rather than free text. Reasoning effort is `low` — this is
short extraction work and p95 latency is a scored metric.

Luna is the default because it is the cheapest and fastest of the GPT-5.6 family
($0.20 / $1.20 per MTok) and saturates this task. `GRIDWISE_MODEL` switches it;
`gpt-5.6-terra` is the step up if the paraphrase suite ever shows misses.

### Semantic conventions the prompt enforces

| Rule | Example |
|---|---|
| Windows are start-inclusive, **end-exclusive**, for *every* connective — to, until, till, through, between, `X-Y` | `6 PM through 9 PM` → `[18, 19, 20]` |
| `factor` is the fraction that **remains**, not the loss | `80% reduction` → `0.2` |
| Relative reserves resolve against `capacity_kwh` | `50% of capacity` on 200 kWh → `100` |
| Ranges may wrap midnight, still ascending | `10 PM until 2 AM` → `[0, 1, 22, 23]` |
| Unrelated notes are `no_op`, never stretched into a rule | `"the cafeteria menu changes"` → `no_op` |

---

## Guardrails

Model output is untrusted structured data. It is **rejected, never repaired**,
when a change would alter meaning:

| Check | Rule |
|---|---|
| Directive type | One of the six supported types |
| Note mapping | Exactly one entry per note, each `note_index` present once, in order |
| Hours | Unique integers 0–23, ascending (deduplicated and sorted) |
| Solar factor | `0 ≤ factor ≤ 1` |
| Battery reserve | Finite, non-negative, not above `capacity_kwh` |
| Grid cap | Finite, non-negative |
| `applies` | **Derived** from the type — `no_op` ⇒ `false` + `null`; every other type ⇒ `true` |

A rejection triggers one retry, budget permitting. If that also fails, or the
model is unreachable, `app/fallback_parser.py` keeps the service answering rather
than returning 5xx. **That parser is a reliability safety net only** — it runs
only after the model path has failed, and it is not the interpretation mechanism
the challenge requires.

---

## Optimizer

Every GridWise rule is linear, so the whole problem is one LP with 96 variables
(`grid`, `solar_used`, `charge`, `discharge` × 24 hours):

- **Objective** — minimize `Σ grid_kwh[h] × tariff[h]`
- **Equalities** — hourly energy balance; total charged = total discharged (end-of-day neutrality)
- **Inequalities** — running state of charge within `[min_energy_after[h], capacity]`
- **Bounds** — grid cap, effective-solar ceiling, per-hour charge/discharge limits, zeroed inside no-charge/no-discharge windows

Three post-processing steps make the solution survive exact replay:

1. **Net out simultaneous charge and discharge.** Degenerate optima can return both non-zero in one hour, but `battery_action` allows only one.
2. **Clamp solar to the effective ceiling**, so floating-point noise cannot look like effective-solar overuse.
3. **Recompute `grid_kwh` from the balance equation** rather than reporting the solver's own value, so the equality holds exactly at the emitted precision.

### Over-constrained scenarios

Organizer scoring scenarios are guaranteed feasible (§5.1), but a misread note —
or a cap below `demand − solar − max_discharge`, which the hourly discharge limit
makes unreachable however full the battery is — can produce an impossible model.

Rather than discarding the directives, `solve_minimum_violation` gives the
numeric ones (grid cap, battery reserve) a heavily penalised slack variable and
re-solves. The result satisfies every hour it **can** and exceeds only where
physics forces it, by the smallest possible margin.

Physical rules are never relaxed — energy balance, battery bounds, rate limits,
effective solar, and the no-charge / no-discharge windows stay hard. Trading
those for a directive would swap a constrained-case violation for a **validity**
failure, and validity is checked on every case.

> Worked example — cap 155 over hours 18–20, with demand 215 and no solar at hour
> 19 and a 50 kWh/h discharge limit, so the floor is 165:
> hour 18 → **155**, hour 19 → **165**, hour 20 → **155**, total violation
> **10.00 kWh**, the physical minimum. A feasible cap yields exactly zero.

---

## Robustness

### Request budget

The judge treats a response beyond 30 s as a failure, so the whole pipeline runs
under one wall-clock budget (`GRIDWISE_REQUEST_BUDGET`, default 20 s). Each
optional step — a guardrail retry, a model escalation — checks the remaining
budget before spending any of it, and each model call's timeout is clamped to
what is left. The worst case is bounded rather than additive.

### Confidence escalation

The interpretation schema carries a `confidence` field. A low-confidence reading
is re-checked once on `gpt-5.6-terra` — but only when the budget can absorb the
second call, since a timed-out response scores zero.

### Ambiguity hedging

Where a time window genuinely supports two readings, the model may return
`alternate_hours` alongside its best reading. The optimizer then constrains the
**union** of both readings, while the response reports only the single best
reading.

This exploits an asymmetry in the scoring. A plan that misses a real directive is
**invalid** — zero for directive application and zero for optimization on that
case. An over-constrained plan is merely slightly more expensive, scoring
`min(1, optimal / ours)`. Hedging converts the first outcome into the second, and
because only `hours` is reported, no interpretation credit is traded away.

It fires only when the model flags ambiguity; across the ten public cases it
never fires, and every case still scores ratio 1.0000. Disable with
`GRIDWISE_HEDGING=0`.

### Interpretation cache

Validated directives are cached in an LRU keyed on the operator notes plus the
battery spec (capacity matters — "50% of capacity" resolves differently on a
different pack). What is stored is the **post-guardrail** directive set, never raw
model output, so a cache hit cannot bypass validation. Judge retries and reruns
cost no model call.

### Observability

Every request carries an id, present on each log line, echoed in the
`x-request-id` response header, and included in error bodies, so a reported
failure maps to a specific log line. Credential-shaped substrings are redacted
from log records before emission, with tests covering OpenAI and Anthropic key
shapes, bearer tokens, and `api_key=` assignments.

---

## API contract

Request and response follow Problem Statement §7 and §10 exactly.

### `GET /health`

```json
{"status": "ok"}
```

Answers immediately and never calls the model, so readiness is not gated on
provider latency.

### `POST /optimize-energy`

| Code | Meaning |
|---|---|
| 200 | Valid schedule returned |
| 400 | Request body is not valid JSON |
| 422 | Well-formed JSON that violates the request schema |
| 500 | Controlled internal error — no stack traces, no secrets |

<details>
<summary>Example response (truncated)</summary>

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Usable solar is reduced to 25% during panel cleaning."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0,
     "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "The plan honours the reduced solar availability, ignores 1 unrelated note, ..."
}
```
</details>

---

## Configuration

All configuration is by environment variable. **No secret is ever committed,
logged, or returned in a response.**

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** | — | OpenAI API credential |
| `GRIDWISE_MODEL` | no | `gpt-5.6-luna` | Model used for note interpretation |
| `GRIDWISE_ESCALATION_MODEL` | no | `gpt-5.6-terra` | Model used when a reading is low-confidence |
| `GRIDWISE_REASONING_EFFORT` | no | `low` | Reasoning effort; empty omits the parameter |
| `GRIDWISE_REQUEST_BUDGET` | no | `20` | Wall-clock seconds for the whole request |
| `GRIDWISE_LLM_TIMEOUT` | no | `12` | Ceiling for one model call, clamped by the budget |
| `GRIDWISE_LLM_MAX_RETRIES` | no | `1` | SDK retries on transient 429/5xx |
| `GRIDWISE_ESCALATION` | no | `1` | `0` disables low-confidence escalation |
| `GRIDWISE_HEDGING` | no | `1` | `0` disables ambiguity hedging |
| `GRIDWISE_CACHE_SIZE` | no | `256` | Interpretation cache entries; `0` disables |
| `GRIDWISE_DISABLE_LLM` | no | `0` | `1` skips the model — offline testing only |
| `PORT` | no | `8000` | Bind port (Render injects this) |

`OPENAI_API_KEY` is supplied one way per environment, never through the repo:

| Environment | How the key is supplied | Committed? |
|---|---|---|
| Local | `.env` file (gitignored, loaded at startup) | No |
| Render | Service → Environment tab; `sync: false` in `render.yaml` stops Render reading it from the repo | No |
| Docker | `-e OPENAI_API_KEY=...` at run time | No — not baked into the image |

---

## Deployment

The repository includes `render.yaml`, so the service can be created as a Blueprint:

1. Render dashboard → **New** → **Blueprint** → select this repository.
2. Render reads `render.yaml` and creates the `gridwise-api` web service.
3. Set **`OPENAI_API_KEY`** in the service's *Environment* tab. It is marked
   `sync: false`, so it is never read from the repository.
4. Verify: `./scripts/smoke_test.sh https://<service>.onrender.com`

Manual setup instead of the blueprint:

- Language / runtime: **Python 3**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

> **Python version matters.** `.python-version` pins 3.12.7. On Python 3.14 the
> pinned `pydantic-core` has no prebuilt wheel, so pip falls back to compiling it
> with Rust and the build fails on Render's read-only filesystem. If you create
> the service manually rather than from the blueprint, also set `PYTHON_VERSION`
> to `3.12.7` in the Environment tab.

Render free instances sleep after ~15 minutes idle and take ~50 s to wake, which
would blow both the health-readiness window and p95 latency. **This deployment is
kept warm by an external cron pinging `/health` every 5 minutes**, comfortably
inside that window.

---

## Docker fallback

**Published image:**

```
imransihab0/gridwise-api:1.0.0
imransihab0/gridwise-api@sha256:4fa1da5e93ded54f39ce0085721a893c65d576921098e6e1aa9471cfffd77b9e
```

Verified anonymously pullable — no credentials needed.

```bash
docker pull imransihab0/gridwise-api:1.0.0
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="sk-proj-..." imransihab0/gridwise-api:1.0.0
curl -s http://127.0.0.1:8000/health     # {"status":"ok"}
```

The image binds `0.0.0.0`, exposes port 8000 (overridable with `-e PORT=...`),
runs as an unprivileged user (uid 10001), pins Python 3.12 to match
`.python-version`, declares a `HEALTHCHECK`, and contains **no baked-in
credentials** — `.dockerignore` keeps `.env` out of the build context entirely,
and the key is supplied at run time.

To rebuild and republish, one script builds, asserts no credential reached the
image, boots it, runs the full smoke test against the container, checks the
`HEALTHCHECK` reaches `healthy`, and prints the digest:

```bash
export OPENAI_API_KEY="sk-proj-..."
./scripts/docker_build.sh                             # local build + verify
./scripts/docker_build.sh imransihab0/gridwise-api    # also pushes
```

---

## Testing

```bash
pip install -r requirements-dev.txt

pytest -q --ignore=tests/test_paraphrases.py   # 112 tests, no API key required
pytest -q                                      # 155 tests, incl. 43 live checks
```

### Against the public sample cases

```bash
python scripts/run_samples.py            # full pipeline, uses the LLM
python scripts/run_samples.py --no-llm   # optimizer only, no API key needed
python scripts/run_samples.py --case SAMPLE-07 -v
```

For each case this compares the interpretation against organizer ground truth,
replays the returned plan against the **ground-truth** directives — exactly as the
judge does, not against our own reading — verifies the reported totals, and scores
cost quality.

```
PASS  SAMPLE-01  Solar cleaning + distractor   cost 38,365.00 vs 38,365.00 (+0.00)  ratio 1.0000  [llm]
...
10/10 cases fully valid
Optimization Quality (10 pts): 10.00
latency  mean 2.62s  p95 3.89s
```

### Paraphrase robustness

`tests/test_paraphrases.py` runs **43 live interpretation checks** covering the
phrasings hidden cases are likely to use: end-exclusive windows across every
connective, remaining-fraction solar factors, percentage- and
fraction-of-capacity reserves, single-hour windows, 24-hour clock without colons,
decimal percentages, MWh units, midnight-wrapping ranges, passive phrasing, and
energy-adjacent distractors that must still be `no_op`. Extend
`tests/paraphrases.json` rather than tuning against the public sample wording.

### Concurrency

```bash
python scripts/load_test.py -c 10 -n 20 --unique     # --unique defeats the cache
python scripts/load_test.py --base https://<service> -c 5 -n 15
```

Measured at concurrency 10 with the cache defeated: 20/20 success, p95 4.2 s, no
serialization.

---

## Project layout

```
app/
  main.py            FastAPI surface, error envelopes, request-id middleware
  service.py         the pipeline: interpret → guardrail → apply → optimize → verify
  llm.py             OpenAI Responses API; the ONLY vendor-aware module
  prompts.py         system prompt and per-request message
  guardrails.py      deterministic validation of model output
  directives.py      validated directives → per-hour constraint arrays
  optimizer.py       the linear program + minimum-violation fallback
  replay.py          judge-equivalent replay, run before responding
  cache.py           LRU over post-guardrail directives
  deadline.py        one wall-clock budget per request
  logging_utils.py   request ids and secret redaction
  fallback_parser.py deterministic safety net for model outages
  config.py          environment-driven settings
  schemas.py         request/response models (Problem Statement §7, §10)
scripts/
  run_samples.py     grade the public cases end to end
  smoke_test.sh      curl checks against any base URL
  load_test.py       concurrency and p95 measurement
  docker_build.sh    build, verify, push, print digest
  healthcheck.py     container health probe
tests/               112 offline + 43 live paraphrase checks
data/                public sample cases
```

Provider isolation is deliberate: swapping vendors means rewriting `app/llm.py`
alone. Prompts, guardrails, directives, optimizer, replay, and every test are
provider-agnostic.

---

## Dependencies

| Package | Role |
|---|---|
| [`fastapi`](https://fastapi.tiangolo.com/) + [`uvicorn`](https://www.uvicorn.org/) | HTTP service and ASGI server |
| [`pydantic`](https://docs.pydantic.dev/) | Request/response schema validation |
| [`openai`](https://github.com/openai/openai-python) | Official OpenAI SDK — operator-note interpretation |
| [`scipy`](https://scipy.org/) (HiGHS) + [`numpy`](https://numpy.org/) | Linear program for the schedule |
| [`python-dotenv`](https://github.com/theskumar/python-dotenv) | Local `.env` loading |
| [`pytest`](https://pytest.org/) + [`httpx`](https://www.python-httpx.org/) | Test suite (dev only) |

Development of this solution used AI coding assistance; the architecture,
optimizer formulation, guardrail design, and test strategy are the team's own.

---

## Known limitations

- **Model dependency.** Interpretation quality is bounded by the model. If the
  OpenAI API is unreachable the deterministic parser keeps the service
  responding, but it handles fewer phrasings, and it does **not** satisfy the
  challenge's LLM requirement — it exists so an outage degrades the score
  instead of zeroing the service.
- **Directive vocabulary is closed.** Only the six specified types are emitted.
  A hidden note describing some other operating condition resolves to `no_op`,
  by design — inventing a directive type is explicitly disallowed.
- **Hedging costs a little optimality when it fires.** By construction it trades
  a small cost increase for validity under either reading. It triggers only on
  model-flagged ambiguity and never fires on the public cases; set
  `GRIDWISE_HEDGING=0` to disable.
- **Escalation is budget-gated.** A low-confidence reading is re-checked on a
  stronger model only if the request budget can absorb the second call. Under a
  slow provider the first reading stands — deliberately, since a timed-out
  response scores zero.
- **Cache is per-process and in-memory.** It does not survive a restart and is
  not shared across instances. Sufficient for single-instance judging; a
  multi-instance deployment would simply see a lower hit rate.
- **Impossible directives are satisfied as far as physics allows, not fully.**
  If a scenario cannot meet a directive, the service returns a physically valid
  plan that violates it by the minimum possible margin rather than returning
  nothing. Organizer scoring scenarios are guaranteed feasible, so this should
  not arise during judging.
