ARG UV_VERSION=0.9.15

FROM python:3.12.13-slim-bookworm AS builder

ARG UV_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN python -m pip install "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock uv.toml README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12.13-slim-bookworm AS runtime

ARG UV_VERSION

LABEL org.opencontainers.image.title="Pricing Recommendation Engine" \
      org.opencontainers.image.description="Constrained dynamic-pricing recommendation API" \
      org.opencontainers.image.source="https://github.com/Miguesh/pricing-recommendation-engine-" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    JOBLIB_TEMP_FOLDER=/tmp/joblib

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --gid 10001 app \
    && adduser --uid 10001 --gid 10001 --disabled-password --gecos "" app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder --chown=10001:10001 /app/pyproject.toml /app/uv.lock /app/model-contract/
RUN test "$(uv --version)" = "uv ${UV_VERSION}"

USER 10001:10001

EXPOSE 8000

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live', timeout=3)" || exit 1

CMD ["uvicorn", "pricing_engine.interfaces.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "100", "--backlog", "2048", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "30", "--no-access-log"]
