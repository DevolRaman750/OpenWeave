# OpenWeave core image: serves the analyze sidecar AND the continuous observer.
# Pick the role at runtime via the command:
#   sidecar:   uvicorn openweave_core.sidecar.app:app --host 0.0.0.0 --port 8000
#   observer:  python -m openweave_core.observer
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code.
COPY openweave_core ./openweave_core
COPY pyproject.toml ./

# Non-root runtime user; writable dirs for cursor + reports.
RUN useradd --create-home --uid 10001 openweave \
    && mkdir -p /data/reports /data/state \
    && chown -R openweave:openweave /app /data
USER openweave

ENV OPENWEAVE_OBSERVE_STATE=/data/state/observer_state.json \
    OPENWEAVE_OBSERVE_OUTPUT_DIR=/data/reports

EXPOSE 8000

# Default role: the analyze sidecar. Override `command` for the observer.
CMD ["uvicorn", "openweave_core.sidecar.app:app", "--host", "0.0.0.0", "--port", "8000"]
