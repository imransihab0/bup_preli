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
from dataclasses import dataclass
from typing import Any, Literal

import openai
from pydantic import BaseModel, Field

from .config import SETTINGS
from .deadline import Deadline
from .prompts import SYSTEM_PROMPT, build_user_message
from .schemas import BatteryInput

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1200

# Seconds a single interpretation call is assumed to need. Used to decide
# whether the budget can still afford an escalation or a retry.
ESTIMATED_CALL_SECONDS = 6.0

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
    confidence: Literal["high", "low"] = Field(
        default="high",
        description=(
            "'low' only when the note is genuinely ambiguous and a different "
            "reading is defensible. Routine notes are 'high'."
        ),
    )
    alternate_hours: list[int] = Field(
        default_factory=list,
        description=(
            "Only when the time window itself is ambiguous: the hours the other "
            "defensible reading would cover. Empty otherwise."
        ),
    )
    explanation: str = Field(description="One short sentence explaining the interpretation.")


class InterpretationBatch(BaseModel):
    interpretations: list[NoteInterpretation]


class LLMUnavailableError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


@dataclass
class InterpretationResult:
    entries: list[dict[str, Any]]
    model: str
    escalated: bool = False


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

    def interpret(
        self,
        notes: list[str],
        battery: BatteryInput,
        deadline: Deadline | None = None,
    ) -> InterpretationResult:
        """Return raw (still untrusted) interpretations, one per note."""
        if not self.enabled:
            raise LLMUnavailableError("LLM interpretation is disabled")

        deadline = deadline or Deadline.start(SETTINGS.request_budget)
        client = self._get_client()
        user_message = build_user_message(notes, battery.model_dump())

        entries = self._call(client, user_message, SETTINGS.model, deadline)
        result = InterpretationResult(entries=entries, model=SETTINGS.model)

        # Escalate only when the model itself flags ambiguity, and only if the
        # budget can absorb a second call. Accuracy is worth more than latency,
        # but not worth a timeout - a timed-out response scores zero.
        if self._should_escalate(entries, deadline):
            logger.info(
                "low-confidence interpretation; escalating to %s", SETTINGS.escalation_model
            )
            try:
                escalated = self._call(
                    client, user_message, SETTINGS.escalation_model, deadline
                )
                result = InterpretationResult(
                    entries=escalated, model=SETTINGS.escalation_model, escalated=True
                )
            except (LLMUnavailableError, *TRANSPORT_ERRORS) as exc:
                # Keep the first answer rather than failing the request.
                logger.warning("escalation failed (%s); keeping primary result",
                               type(exc).__name__)
        return result

    @staticmethod
    def _should_escalate(entries: list[dict[str, Any]], deadline: Deadline) -> bool:
        if not SETTINGS.escalation_enabled:
            return False
        if not deadline.allows(ESTIMATED_CALL_SECONDS):
            logger.info("skipping escalation: %.1fs left in budget", deadline.remaining)
            return False
        return any(entry.get("confidence") == "low" for entry in entries)

    # -- transport ---------------------------------------------------------- #
    def _call(
        self, client: openai.OpenAI, user_message: str, model: str, deadline: Deadline
    ) -> list[dict[str, Any]]:
        timeout = deadline.timeout_for(SETTINGS.llm_timeout, reserve=1.0)
        if timeout <= 0:
            raise LLMUnavailableError("request budget exhausted before the model call")
        scoped = client.with_options(timeout=timeout)

        try:
            return self._parse_call(scoped, user_message, model)
        except TRANSPORT_ERRORS as exc:
            raise LLMUnavailableError(f"model request failed: {type(exc).__name__}") from exc
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning(
                "structured output path failed (%s); retrying as raw JSON", type(exc).__name__
            )
            try:
                return self._json_call(scoped, user_message, model)
            except TRANSPORT_ERRORS as inner:
                raise LLMUnavailableError(
                    f"model request failed: {type(inner).__name__}"
                ) from inner

    def _request_kwargs(self) -> dict[str, Any]:
        """Reasoning effort is kept low: this is short extraction, and p95
        latency is a scored metric. Models that reject the parameter are
        detected once and then called without it."""
        if self._reasoning_supported and SETTINGS.reasoning_effort:
            return {"reasoning": {"effort": SETTINGS.reasoning_effort}}
        return {}

    def _parse_call(
        self, client: openai.OpenAI, user_message: str, model: str
    ) -> list[dict[str, Any]]:
        """Preferred path: schema-constrained output validated by the SDK."""
        try:
            response = client.responses.parse(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=user_message,
                text_format=InterpretationBatch,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                **self._request_kwargs(),
            )
        except openai.BadRequestError as exc:
            if self._reasoning_supported and "reasoning" in str(exc).lower():
                logger.info("model %s rejects the reasoning parameter; disabling it", model)
                self._reasoning_supported = False
                return self._parse_call(client, user_message, model)
            raise

        parsed = response.output_parsed
        if parsed is None:
            raise LLMUnavailableError("model returned no parsable interpretation")
        return [item.model_dump() for item in parsed.interpretations]

    def _json_call(
        self, client: openai.OpenAI, user_message: str, model: str
    ) -> list[dict[str, Any]]:
        """Fallback path for models or SDK versions without structured outputs."""
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=(
                f"{user_message}\n\n"
                'Reply with JSON only: {"interpretations": [ ... ]}. Each item has '
                "note_index, directive_type, hours, factor, minimum_energy_kwh, "
                "max_grid_kwh, confidence, alternate_hours, explanation. Use null "
                "for fields that do not apply."
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
