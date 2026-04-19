# INA Speech Segmenter API

A lightweight FastAPI wrapper around
[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
that accepts audio file uploads and returns labeled time segments
(speech, music, noise, etc.). The service is designed to run in Docker on
CPU via TensorFlow.

## Service

- Container: `ina-segmenter-api-ina-segmenter-1`
- Host port: `8002`
- Container port: `8000`
- Endpoint: `POST /segment`
- Model: loaded once at startup with `detect_gender=False`
- Model cache: persisted in a Docker volume mounted at `/root/.cache`

## Run

```bash
docker compose up --build -d
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
