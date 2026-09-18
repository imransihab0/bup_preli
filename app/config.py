"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present, for development convenience. Real deployments
# (Render, Docker) inject environment variables directly and have no .env file;
# `override=False` means a genuine environment variable always wins.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Judge tolerance from the Problem Statement (S11.5): 0.01 kWh / 0.01 BDT.
# We hold ourselves to a tighter internal bar so rounding never eats the margin.
JUDGE_TOLERANCE = 0.01
INTERNAL_TOLERANCE = 1e-6

# Decimal places used when emitting plan numbers. Well inside JUDGE_TOLERANCE.
OUTPUT_PRECISION = 6


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    escalation_model: str
    reasoning_effort: str
    llm_timeout: float
    llm_max_retries: int
    disable_llm: bool
    request_budget: float
    escalation_enabled: bool
    hedging_enabled: bool
    cache_size: int

    @property
    def llm_enabled(self) -> bool:
        return not self.disable_llm and bool(self.api_key)


def load_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("OPENAI_API_KEY") or None,
        model=os.environ.get("GRIDWISE_MODEL") or "gpt-5.6-luna",
        # Empty string omits the parameter entirely.
        escalation_model=os.environ.get("GRIDWISE_ESCALATION_MODEL") or "gpt-5.6-terra",
        reasoning_effort=os.environ.get("GRIDWISE_REASONING_EFFORT", "low"),
        llm_timeout=_env_float("GRIDWISE_LLM_TIMEOUT", 12.0),
        llm_max_retries=_env_int("GRIDWISE_LLM_MAX_RETRIES", 1),
        disable_llm=os.environ.get("GRIDWISE_DISABLE_LLM", "0") == "1",
        # Total wall-clock budget for one request. The judge fails anything over
        # 30 s, so this leaves comfortable headroom for the optimizer and I/O.
        request_budget=_env_float("GRIDWISE_REQUEST_BUDGET", 20.0),
        escalation_enabled=os.environ.get("GRIDWISE_ESCALATION", "1") == "1",
        hedging_enabled=os.environ.get("GRIDWISE_HEDGING", "1") == "1",
        cache_size=_env_int("GRIDWISE_CACHE_SIZE", 256),
    )


SETTINGS = load_settings()
