# GridWise — LLM-Assisted Smart Campus Energy Optimization

BUP CSE Fest 2026 Hackathon · Online Preliminary · `POST /optimize-energy`

A deployed HTTP service that reads free-text campus operator notes, converts them
into machine-checkable directives with an LLM, validates those directives
deterministically, and returns a cost-minimal 24-hour energy schedule that obeys
them.

---

## Quickstart (clean machine → working service in ~2 minutes)

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone <this-repository-url>
cd bup_preli

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key on the OPENAI_API_KEY line
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`.env` is gitignored and loaded automatically at startup. A real environment
variable always takes precedence over it, so deployments that inject
`OPENAI_API_KEY` directly (Render, Docker) need no `.env` file at all.

Verify in a second terminal:

```bash
# 1. readiness
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}

# 2. one public sample case end to end
curl -s -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  --data @<(python3 -c "import json;print(json.dumps(json.load(open('data/public_sample_cases.json'))['cases'][0]['input']))")
```

Expected: HTTP 200 with `scenario_id: "SAMPLE-01"`, a 24-entry `hourly_plan`,
`total_cost_bdt` of **38365.00**, and two `directive_interpretation` entries —
`solar_reduction` over hours `[12, 13]` with `factor 0.25`, and a `no_op`.

Or run both checks plus the error-code cases at once:

```bash
./scripts/smoke_test.sh                            # local
./scripts/smoke_test.sh https://<your-service>     # deployed
```

---

## Testing against the public sample cases

```bash
python scripts/run_samples.py            # full pipeline, uses the LLM
python scripts/run_samples.py --no-llm   # optimizer only, no API key needed
python scripts/run_samples.py --case SAMPLE-07 -v
```

For each case this reports the interpretation against organizer ground truth,
replays the returned plan against the **ground-truth** directives (exactly as the
judge does, not against our own reading), verifies the reported totals, and
scores cost quality.

Expected result — all ten cases valid at the organizer optimum:

```
PASS  SAMPLE-01  Solar cleaning + distractor     cost 38,365.00 vs 38,365.00 (+0.00)  ratio 1.0000
...
10/10 cases fully valid
Optimization Quality (10 pts): 10.00
```

Unit and contract tests:

```bash
pip install -r requirements-dev.txt
pytest -q                                # 57 tests, no API key required
pytest tests/test_paraphrases.py -v      # 22 live paraphrase checks, needs a key
```

---

## Architecture

```
 operator_notes (natural language)
        │
        ▼
 ┌──────────────────┐   The model interprets every note into a flat
 │  app/llm.py      │   structured directive. One call per request covers all 1-3 notes.
 └──────────────────┘   Model output is treated as UNTRUSTED from here on.
        │
        ▼
 ┌──────────────────┐   Deterministic validation: allowed types only, one entry
 │ app/guardrails.py│   per note in order, hours unique ints 0-23 ascending,
 └──────────────────┘   factor in [0,1], reserve <= capacity, cap >= 0.
        │               `applies` is DERIVED here, never taken from the model.
        ▼
 ┌──────────────────┐   Directives fold into per-hour constraint arrays:
 │ app/directives.py│   effective solar, charge/discharge windows, energy
 └──────────────────┘   floor, grid cap. Overlaps resolve to the tighter bound.
        │
        ▼
 ┌──────────────────┐   96-variable linear program solved with HiGHS.
 │ app/optimizer.py │   Exact optimum — no heuristics, no search.
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   The finished plan is replayed against every judge rule
 │  app/replay.py   │   BEFORE it is returned. A plan that fails replay is
 └──────────────────┘   never shipped.
        │
        ▼
     JSON response
```

`app/service.py` wires these together; `app/main.py` is the FastAPI surface.

### The LLM's role

The model performs the operator-note interpretation itself — it decides whether a
note applies, which directive type it is, which hours it covers, and what the
numeric value is. That structured output is what the optimizer's constraints are
built from. The model is not used for `plan_summary` or cosmetic text; the
summary is generated deterministically from the directives that were applied.

**Model:** `gpt-5.6-luna` via the official `openai` Python SDK
(`client.responses.parse` with a Pydantic schema, so the response is
schema-constrained rather than free text). Reasoning effort is set to `low` —
this is short extraction work and p95 latency is a scored metric.

Configurable with `GRIDWISE_MODEL` and `GRIDWISE_REASONING_EFFORT`. Luna is the
default because it is the cheapest and fastest of the GPT-5.6 family
($0.20/$1.20 per MTok) and this task is well within its capability; switch to
`gpt-5.6-terra` if the paraphrase suite shows misses.

### Guardrails

Model output is rejected, never repaired, when it would change meaning:

| Check | Rule |
|---|---|
| Directive type | Must be one of the six supported types |
| Note mapping | Exactly one entry per note, each `note_index` present once, in order |
| Hours | Unique integers 0–23, returned ascending (deduplicated and sorted) |
| Solar factor | `0 ≤ factor ≤ 1` |
| Battery reserve | Finite, non-negative, not above `capacity_kwh` |
| Grid cap | Finite, non-negative |
| `applies` | Derived from the type — `no_op` ⇒ `false` + `null` adjustment; every other type ⇒ `true` |

A guardrail rejection triggers one retry. If that also fails, or the model is
unreachable, a deterministic parser (`app/fallback_parser.py`) keeps the service
answering instead of returning 5xx. **That parser is a reliability safety net
only** — it runs only when the model path has already failed, and it is not the
interpretation mechanism the challenge requires.

### Optimizer

Every GridWise rule is linear, so the whole problem is one LP with 96 variables
(`grid`, `solar_used`, `charge`, `discharge` × 24 hours):

- **Objective** — minimize `Σ grid_kwh[h] × tariff[h]`
- **Equalities** — hourly energy balance; total charged = total discharged (end-of-day neutrality)
- **Inequalities** — running state of charge within `[min_energy_after[h], capacity]`
- **Bounds** — grid cap, effective solar ceiling, per-hour charge/discharge limits, zeroed inside no-charge/no-discharge windows

Three post-processing steps make the solution survive exact replay: simultaneous
charge and discharge in one hour are netted to a single `battery_action`, solar
is clamped to the effective ceiling, and `grid_kwh` is recomputed from the
balance equation rather than read off the solver, so the equality holds exactly
at the emitted precision.

Verified: the LP reproduces the organizer's reference cost on all ten public
cases to the cent.

---

## API

### `GET /health`

```json
{"status": "ok"}
```

Answers immediately and never calls the model, so readiness is not gated on
provider latency.

### `POST /optimize-energy`

Request and response follow Problem Statement §7 and §10 exactly.

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
  "plan_summary": "Honours the reduced solar availability, ignores 1 unrelated note, ..."
}
```
</details>

---

## Configuration

All configuration is by environment variable. **No secret is ever committed, logged, or returned in a response.**

`OPENAI_API_KEY` is supplied in exactly one way per environment, and never
through the repository:

| Environment | How the key is supplied | Committed? |
|---|---|---|
| Local development | `.env` file (gitignored, loaded at startup) | No |
| Render | Service → Environment tab; `sync: false` in `render.yaml` stops Render reading it from the repo | No |
| Docker | `-e OPENAI_API_KEY=...` at run time | No — not baked into the image |

Error responses are checked by test for absence of stack traces, vendor names,
and key material.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** | — | OpenAI API credential |
| `GRIDWISE_MODEL` | no | `gpt-5.6-luna` | Model used for note interpretation |
| `GRIDWISE_REASONING_EFFORT` | no | `low` | Reasoning effort; empty omits the parameter |
| `GRIDWISE_LLM_TIMEOUT` | no | `12` | Seconds before a model call is abandoned |
| `GRIDWISE_LLM_MAX_RETRIES` | no | `1` | SDK retries on transient 429/5xx |
| `GRIDWISE_DISABLE_LLM` | no | `0` | `1` skips the model — offline testing only |
| `PORT` | no | `8000` | Bind port (Render injects this) |

---

## Deployment (Render)

The repository includes `render.yaml`, so the service can be created as a Blueprint:

1. Render dashboard → **New** → **Blueprint** → select this repository.
2. Render reads `render.yaml` and creates the `gridwise-api` web service.
3. Set **`OPENAI_API_KEY`** in the service's *Environment* tab. It is marked
   `sync: false` in the blueprint, so it is never read from the repository.
4. Deploy, then verify: `./scripts/smoke_test.sh https://<service>.onrender.com`

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

> **Free-tier note:** Render's free instances sleep after inactivity and can take
> ~50 s to wake. Hit `/health` shortly before judging begins, or use a paid
> instance, so the first hidden request is not a cold start.

### Docker fallback

```bash
docker build -t gridwise-api:1.0.0 .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="sk-proj-..." gridwise-api:1.0.0
curl -s http://127.0.0.1:8000/health     # {"status":"ok"}
```

The image binds `0.0.0.0`, exposes port 8000 (overridable with `-e PORT=...`),
runs as an unprivileged user, and contains **no baked-in credentials** — the key
is supplied at run time.

---

## Dependencies

| Package | Role |
|---|---|
| [`fastapi`](https://fastapi.tiangolo.com/) + [`uvicorn`](https://www.uvicorn.org/) | HTTP service and ASGI server |
| [`pydantic`](https://docs.pydantic.dev/) | Request/response schema validation |
| [`openai`](https://github.com/openai/openai-python) | Official OpenAI SDK — operator-note interpretation |
| [`scipy`](https://scipy.org/) (HiGHS) + [`numpy`](https://numpy.org/) | Linear program for the schedule |
| [`pytest`](https://pytest.org/) | Test suite (dev only) |

Development of this solution used AI coding assistance; the architecture,
optimizer formulation, guardrail design, and test strategy are the team's own.

---

## Known limitations

- **Free-tier cold starts.** See the Render note above — the first request after
  idling can be slow. Not a code path, but it affects measured latency.
- **Model dependency.** Interpretation quality is bounded by the model. If the
  OpenAI API is unreachable the deterministic parser keeps the service
  responding, but it handles fewer phrasings than the model does.
- **Directive vocabulary is closed.** Only the six specified types are emitted.
  A hidden note describing some other operating condition resolves to `no_op`,
  by design — inventing a directive type is explicitly disallowed.
- **Infeasible interpretations.** If a misread note produces mutually
  unsatisfiable constraints, the service relaxes the directive layer and returns
  a physically valid plan rather than nothing. Organizer scoring scenarios are
  guaranteed feasible, so this should not trigger during judging.
- **No persistence.** Each request is independent; nothing is cached between
  requests, so identical repeat scenarios pay the model call again.
