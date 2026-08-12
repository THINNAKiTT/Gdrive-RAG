FROM python:3.13-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra all --no-install-project
COPY . .

RUN uv sync --extra all
RUN mkdir -p /app/chroma_db /app/logs /app/config