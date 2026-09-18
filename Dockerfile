# Fallback execution path for organizers.
#
# No secrets are baked in: OPENAI_API_KEY is supplied at run time with
# `docker run -e`. .dockerignore additionally keeps .env out of the build
# context entirely, so it cannot be copied in by accident.
#
# Python is pinned to 3.12 to match .python-version. On 3.14 the pinned
# pydantic-core has no prebuilt wheel and pip falls back to a Rust build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Dependencies first so image rebuilds reuse the layer. The generous retry and
# timeout settings keep the build from dying on a slow or flaky link to PyPI -
# a plain `pip install` aborts on the first read timeout and loses the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Run unprivileged.
RUN useradd --create-home --uid 10001 gridwise && chown -R gridwise:gridwise /app
USER gridwise

EXPOSE 8000

COPY --chown=gridwise:gridwise scripts/healthcheck.py /app/healthcheck.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

# Shell form so $PORT expands; Render injects it, local runs default to 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
