FROM python:3.14.7-slim-bookworm

ARG MLFLOW_VERSION=3.14.0
ARG PSYCOPG2_VERSION=2.9.12
ARG BOTO3_VERSION=1.43.51

LABEL org.opencontainers.image.title="Pricing Engine MLflow Server" \
      org.opencontainers.image.description="Local MLflow registry backed by PostgreSQL and S3-compatible storage" \
      org.opencontainers.image.source="https://github.com/Miguesh/pricing-recommendation-engine-" \
      org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

RUN python -m pip install \
      "mlflow==${MLFLOW_VERSION}" \
      "psycopg2-binary==${PSYCOPG2_VERSION}" \
      "boto3==${BOTO3_VERSION}" \
    && python -m pip check \
    && addgroup --gid 10002 mlflow \
    && adduser --uid 10002 --gid 10002 --disabled-password --gecos "" mlflow

USER 10002:10002

EXPOSE 5000

STOPSIGNAL SIGTERM
