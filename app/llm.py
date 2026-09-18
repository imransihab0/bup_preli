"""OpenAI-backed interpretation of operator notes.

This is the mandatory language-model step (Problem Statement S02). The model
produces the structured interpretation; `guardrails` then validates it before
anything reaches the optimizer. The model never sees the schedule and never
decides numbers that were not in the note.

Provider isolation: this module is the ONLY place that knows which vendor SDK is
in use. Prompts, guardrails, directives, optimizer, and tests are all
provider-agnostic, so swapping vendors means rewriting this file alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

import openai
from pydantic import BaseModel, Field

from .config import SETTINGS
from .prompts import SYSTEM_PROMPT, build_user_message
from .schemas import BatteryInput

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1200

# Errors worth surfacing as "model unavailable" so the caller can degrade.
TRANSPORT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.InternalServerError,
    openai.APIStatusError,
)


class NoteInterpretation(BaseModel):
    """Flat shape on purpose - it is markedly more reliable to generate than a
    nested optional object, and `guardrails` rebuilds the nested
    `structured_adjustment` the judge expects."""

    note_index: int = Field(description="Zero-based index of the operator note.")
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    hours: list[int] = Field(
        default_factory=list,
        description="Affected hours 0-23, ascending. Empty for no_op.",
    )
    factor: float | None = Field(
        default=None, description="solar_reduction only: usable fraction remaining, 0-1."
    )
    minimum_energy_kwh: float | None = Field(
        default=None, description="minimum_battery_reserve only: absolute kWh floor."
    )
    max_grid_kwh: float | None = Field(
        default=None, description="max_grid_window only: per-hour grid import cap in kWh."
    )
    explanation: str = Field(description="One short sentence explaining the interpretation.")


class InterpretationBatch(BaseModel):
    interpretations: list[NoteInterpretation]


class LLMUnavailableError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


class NoteInterpreter:
    """Thin wrapper over the Responses API. One call covers all notes at once,
    which keeps p95 latency inside the judge's 5 s band."""

    def __init__(self) -> None:
        self._client: openai.OpenAI | None = None
        self._reasoning_supported = True

    @property
    def enabled(self) -> bool:
        return SETTINGS.llm_enabled

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            if not SETTINGS.api_key:
                raise LLMUnavailableError("OPENAI_API_KEY is not configured")
            self._client = openai.OpenAI(
                api_key=SETTINGS.api_key,
                timeout=SETTINGS.llm_timeout,
                max_retries=SETTINGS.llm_max_retries,
            )
        return self._client

    def interpret(self, notes: list[str], battery: BatteryInput) -> list[dict[str, Any]]:
        """Return raw (still untrusted) interpretation dicts, one per note."""
        if not self.enabled:
            raise LLMUnavailableError("LLM interpretation is disabled")

        client = self._get_client()
        user_message = build_user_message(notes, battery.model_dump())

        try:
            return self._parse_call(client, user_message)
        except TRANSPORT_ERRORS as exc:
            raise LLMUnavailableError(f"model request failed: {type(exc).__name__}") from exc
        except LLMUnavailableError:
            raise
        except Exception as exc:
            # Structured-output path unsupported or malformed: try plain JSON.
            logger.warning(
                "structured output path failed (%s); retrying as raw JSON", type(exc).__name__
            )
            try:
                return self._json_call(client, user_message)
            except TRANSPORT_ERRORS as inner:
                raise LLMUnavailableError(
                    f"model request failed: {type(inner).__name__}"
                ) from inner

    # -- transport variants ------------------------------------------------- #
    def _request_kwargs(self) -> dict[str, Any]:
        """Reasoning effort is kept low: this is short extraction, and p95
        latency is a scored metric. Models that reject the parameter are
        detected once and then called without it."""
        if self._reasoning_supported and SETTINGS.reasoning_effort:
            return {"reasoning": {"effort": SETTINGS.reasoning_effort}}
        return {}

    def _parse_call(self, client: openai.OpenAI, user_message: str) -> list[dict[str, Any]]:
        """Preferred path: schema-constrained output validated by the SDK."""
        try:
            response = client.responses.parse(
                model=SETTINGS.model,
                instructions=SYSTEM_PROMPT,
                input=user_message,
                text_format=InterpretationBatch,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                **self._request_kwargs(),
            )
        except openai.BadRequestError as exc:
            if self._reasoning_supported and "reasoning" in str(exc).lower():
                logger.info("model %s rejects the reasoning parameter; disabling it",
                            SETTINGS.model)
                self._reasoning_supported = False
                return self._parse_call(client, user_message)
            raise

        parsed = response.output_parsed
        if parsed is None:
            raise LLMUnavailableError("model returned no parsable interpretation")
        return [item.model_dump() for item in parsed.interpretations]

    def _json_call(self, client: openai.OpenAI, user_message: str) -> list[dict[str, Any]]:
        """Fallback path for models or SDK versions without structured outputs."""
        response = client.responses.create(
            model=SETTINGS.model,
            instructions=SYSTEM_PROMPT,
            input=(
                f"{user_message}\n\n"
                'Reply with JSON only: {"interpretations": [ ... ]}. Each item has '
                "note_index, directive_type, hours, factor, minimum_energy_kwh, "
                "max_grid_kwh, explanation. Use null for fields that do not apply."
            ),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            **self._request_kwargs(),
        )
        return _extract_interpretations(response.output_text or "")


def _extract_interpretations(text: str) -> list[dict[str, Any]]:
    """Pull the interpretation list out of a raw text response.

    Tolerates a fenced code block or surrounding prose, but never guesses at
    missing fields - anything unparsable raises so the caller can fall back.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMUnavailableError("model response contained no JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError("model response was not valid JSON") from exc

    if isinstance(payload, dict):
        entries = payload.get("interpretations", payload.get("directive_interpretation"))
    else:
        entries = payload
    if not isinstance(entries, list):
        raise LLMUnavailableError("model response did not contain an interpretation list")
    return entries


INTERPRETER = NoteInterpreter()
