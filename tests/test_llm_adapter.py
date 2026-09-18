"""Provider-adapter behaviour, exercised without network access.

The live model is covered by tests/test_paraphrases.py; this file pins the
plumbing around it - JSON recovery, transport-error translation, and the
reasoning-parameter fallback.
"""

from __future__ import annotations

from dataclasses import replace

import httpx2
import openai
import pytest

from app.deadline import Deadline
from app.llm import (
    InterpretationBatch,
    LLMUnavailableError,
    NoteInterpretation,
    NoteInterpreter,
    _extract_interpretations,
)

VALID = '{"interpretations": [{"note_index": 0, "directive_type": "no_op", "explanation": "x"}]}'


class TestJSONRecovery:
    def test_plain_object(self):
        assert _extract_interpretations(VALID)[0]["directive_type"] == "no_op"

    def test_fenced_code_block(self):
        assert len(_extract_interpretations(f"```json\n{VALID}\n```")) == 1

    def test_surrounding_prose(self):
        assert len(_extract_interpretations(f"Here you go:\n{VALID}\nHope that helps!")) == 1

    def test_directive_interpretation_key_also_accepted(self):
        text = '{"directive_interpretation": [{"note_index": 0, "directive_type": "no_op"}]}'
        assert len(_extract_interpretations(text)) == 1

    @pytest.mark.parametrize("text", ["", "no json here", "{broken", "{}", '{"other": 1}'])
    def test_unusable_text_raises(self, text):
        with pytest.raises(LLMUnavailableError):
            _extract_interpretations(text)


class _StubResponses:
    """Minimal stand-in for client.responses."""

    def __init__(self, *, parsed=None, raise_with=None, text=None):
        self.parsed, self.raise_with, self.text = parsed, raise_with, text
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_with is not None:
            error, self.raise_with = self.raise_with, None
            raise error
        return type("R", (), {"output_parsed": self.parsed})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"output_text": self.text})()


class _StubClient:
    def __init__(self, responses):
        self.responses = responses

    def with_options(self, **_kwargs):
        # The adapter scopes each call's timeout to the remaining budget.
        return self


def _interpreter(responses) -> NoteInterpreter:
    interpreter = NoteInterpreter()
    interpreter._client = _StubClient(responses)
    return interpreter


def _bad_request(message: str) -> openai.BadRequestError:
    # openai 3.x is built on httpx2, not httpx.
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=None)


@pytest.fixture(autouse=True)
def _enable_llm(monkeypatch):
    """Settings is a frozen dataclass, so swap the whole object."""
    from app import llm

    monkeypatch.setattr(
        llm, "SETTINGS", replace(llm.SETTINGS, api_key="test-key", disable_llm=False)
    )


class TestAdapter:
    def test_structured_path_returns_dicts(self, battery):
        batch = InterpretationBatch(
            interpretations=[
                NoteInterpretation(
                    note_index=0, directive_type="no_charge_window", hours=[2, 3], explanation="x"
                )
            ]
        )
        interpreter = _interpreter(_StubResponses(parsed=batch))
        result = interpreter.interpret(["charger offline 2-4 AM"], battery)
        assert result.entries[0]["directive_type"] == "no_charge_window"
        assert result.entries[0]["hours"] == [2, 3]
        assert result.escalated is False

    def test_reasoning_effort_is_sent(self, battery):
        batch = InterpretationBatch(interpretations=[])
        responses = _StubResponses(parsed=batch)
        _interpreter(responses).interpret(["note"], battery)
        assert responses.calls[0]["reasoning"] == {"effort": "low"}

    def test_model_rejecting_reasoning_is_retried_without_it(self, battery):
        batch = InterpretationBatch(interpretations=[])
        responses = _StubResponses(
            parsed=batch, raise_with=_bad_request("Unsupported parameter: 'reasoning'")
        )
        interpreter = _interpreter(responses)
        interpreter.interpret(["note"], battery)
        assert "reasoning" in responses.calls[0]
        assert "reasoning" not in responses.calls[1]
        assert interpreter._reasoning_supported is False

    def test_missing_parsed_output_raises(self, battery):
        interpreter = _interpreter(_StubResponses(parsed=None))
        with pytest.raises(LLMUnavailableError):
            interpreter.interpret(["note"], battery)

    def test_transport_error_becomes_llm_unavailable(self, battery):
        error = openai.APIConnectionError(request=httpx2.Request("POST", "https://x"))
        interpreter = _interpreter(_StubResponses(raise_with=error))
        with pytest.raises(LLMUnavailableError):
            interpreter.interpret(["note"], battery)

    def test_disabled_interpreter_raises(self, battery, monkeypatch):
        from app import llm

        monkeypatch.setattr(llm, "SETTINGS", replace(llm.SETTINGS, disable_llm=True))
        with pytest.raises(LLMUnavailableError):
            NoteInterpreter().interpret(["note"], battery)


@pytest.fixture
def battery():
    from app.schemas import BatteryInput

    return BatteryInput(
        capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40,
        max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
    )


class TestBudgetAndEscalation:
    def test_expired_budget_refuses_to_call(self, battery):
        responses = _StubResponses(parsed=InterpretationBatch(interpretations=[]))
        interpreter = _interpreter(responses)
        spent = Deadline(budget=10.0, started=-1e9)  # already long past
        with pytest.raises(LLMUnavailableError):
            interpreter.interpret(["note"], battery, spent)
        assert responses.calls == []

    def test_low_confidence_triggers_escalation(self, battery, monkeypatch):
        from app import llm

        entry = NoteInterpretation(
            note_index=0, directive_type="no_charge_window", hours=[2],
            confidence="low", explanation="ambiguous",
        )
        responses = _StubResponses(parsed=InterpretationBatch(interpretations=[entry]))
        result = _interpreter(responses).interpret(["note"], battery)
        assert result.escalated is True
        assert result.model == llm.SETTINGS.escalation_model
        assert len(responses.calls) == 2

    def test_high_confidence_makes_one_call(self, battery):
        entry = NoteInterpretation(
            note_index=0, directive_type="no_op", confidence="high", explanation="x",
        )
        responses = _StubResponses(parsed=InterpretationBatch(interpretations=[entry]))
        result = _interpreter(responses).interpret(["note"], battery)
        assert result.escalated is False
        assert len(responses.calls) == 1

    def test_escalation_skipped_when_budget_is_short(self, battery):
        entry = NoteInterpretation(
            note_index=0, directive_type="no_op", confidence="low", explanation="x",
        )
        responses = _StubResponses(parsed=InterpretationBatch(interpretations=[entry]))
        # Enough budget for the first call, not for a second.
        tight = Deadline(budget=4.0, started=Deadline.start(0).started)
        result = _interpreter(responses).interpret(["note"], battery, tight)
        assert result.escalated is False
        assert len(responses.calls) == 1
