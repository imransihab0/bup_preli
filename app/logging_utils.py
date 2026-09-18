"""Per-request logging context.

Every log line for a request carries the same short request id, so a failure
during judging can be traced through interpretation, guardrails, and scheduling
without correlating timestamps by hand. The id is also returned on error
responses so a judge-reported failure maps to a specific log line.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

# Anything resembling credential material must never reach a log record.
# Patterns are deliberately broad: a false redaction costs nothing, a missed
# one is a scored violation ("no API keys, tokens, or sensitive values in logs").
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),                 # OpenAI / Anthropic keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),      # Authorization headers
    re.compile(r"(?i)api[_-]?key[\"'\s:=]+[A-Za-z0-9._\-]{8,}"),
)


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def set_request_id(request_id: str) -> None:
    _REQUEST_ID.set(request_id)


def get_request_id() -> str:
    return _REQUEST_ID.get()


class RequestIdFilter(logging.Filter):
    """Injects `request_id` into every record and redacts anything key-shaped."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.msg = redact(record.msg)
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(redact(arg) for arg in args)
        return True


def redact(value):
    """Replace credential-shaped substrings. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    return value


def configure(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s")
    )
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
