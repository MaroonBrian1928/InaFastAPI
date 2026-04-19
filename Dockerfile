FROM python:3.13-slim

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml .
RUN uv sync --no-dev
RUN uv pip install --python .venv/bin/python --no-deps inaSpeechSegmenter
COPY main.py .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
