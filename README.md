# INA Speech Segmenter API

A lightweight FastAPI wrapper around
[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
that accepts audio file uploads and returns labeled time segments
(speech, music, noise, etc.). The service is designed to run in Docker on
CPU via TensorFlow. Local development uses `mise` for tool management and
`uv` for Python dependency management and command execution.

## Service

- Container: `ina-segmenter-api-ina-segmenter-1`
- Host port: `8002`
- Container port: `8000`
- Endpoint: `POST /segment`
- Model: loaded on demand in a dedicated worker process with `detect_gender=False`
- Worker lifecycle: kept warm while requests are active, then terminated after the idle timeout to release TensorFlow memory back to the OS
- Model cache: persisted in a Docker volume mounted at `/root/.cache`

## Run

```bash
docker compose up --build -d
```

## Local Development

```bash
mise install
mise run install
mise run run
```

## Usage

```bash
curl -X POST http://localhost:8002/segment \
  -F "file=@episode.mp3"
```

Example response:

```json
[
  {"label": "speech", "start": 0.0, "end": 12.5},
  {"label": "music", "start": 12.5, "end": 18.3},
  {"label": "speech", "start": 18.3, "end": 45.1}
]
```

## Segment Labels

INA commonly emits labels such as `speech`, `music`, `noise`, and
`noEnergy`, identifying the dominant audio content in each detected
time window.
