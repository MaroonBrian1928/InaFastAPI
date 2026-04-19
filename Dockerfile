FROM python:3.11-slim

ENV TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflow-cpu && \
    pip install --no-cache-dir --no-deps inaSpeechSegmenter && \
    pip install --no-cache-dir fastapi uvicorn python-multipart pyannote.core sortedcontainers pytextgrid soundfile scikit-image matplotlib

WORKDIR /app
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
