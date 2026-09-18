"""FastAPI application exposing the two judged endpoints (Problem Statement S06)."""

from __future__ import annotations

import logging
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import SETTINGS
from .schemas import OptimizeRequest, OptimizeResponse
from .service import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gridwise")

# Starlette renamed the 422 constant; the numeric code is stable across versions.
HTTP_422 = 422


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "GridWise ready | model=%s | llm_enabled=%s | timeout=%.1fs",
        SETTINGS.model,
        SETTINGS.llm_enabled,
        SETTINGS.llm_timeout,
    )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="GridWise - LLM-Assisted Smart Campus Energy Optimization",
    description=(
        "BUP CSE Fest 2026 preliminary. Interprets natural-language operator notes "
        "with an LLM, validates them deterministically, and returns a cost-minimal "
        "24-hour energy schedule."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe. Deliberately does not call the model - the judge polls
    this before hidden tests begin and it must answer immediately."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(payload: dict) -> JSONResponse:
    started = time.perf_counter()

    # Parsed manually so a schema violation is a clean 422 rather than FastAPI's
    # default envelope, and so nothing about the request leaks into the response.
    try:
        request = OptimizeRequest.model_validate(payload)
    except ValidationError as exc:
        logger.info("rejected invalid request: %d validation error(s)", exc.error_count())
        return JSONResponse(
            status_code=HTTP_422,
            content={
                "error": "invalid_request",
                "detail": "Request does not match the required schema.",
                "violations": _summarize_violations(exc),
            },
        )

    try:
        outcome = run(request)
    except Exception:
        # Controlled failure: never surface a stack trace or model detail (S06.1).
        logger.exception("optimization failed for scenario %s", request.scenario_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_error", "detail": "Failed to produce a schedule."},
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "scenario=%s interpreter=%s degraded=%s cost=%.2f latency_ms=%.0f",
        request.scenario_id,
        outcome.interpreter,
        outcome.degraded,
        outcome.response.total_cost_bdt,
        elapsed_ms,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=outcome.response.model_dump())


@app.exception_handler(RequestValidationError)
async def malformed_body(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Unparsable JSON body -> 400 (S06.1)."""
    logger.info("rejected malformed request body")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "malformed_request", "detail": "Request body is not valid JSON."},
    )


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "Unexpected server error."},
    )


def _summarize_violations(exc: ValidationError, limit: int = 8) -> list[str]:
    return [
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors()[:limit]
    ]
