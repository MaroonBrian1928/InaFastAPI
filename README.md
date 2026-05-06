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
- Long uploads: split into bounded audio chunks before segmentation to reduce peak TensorFlow memory
- Worker lifecycle: kept warm while requests are active, then terminated after the idle timeout to release TensorFlow memory back to the OS
- Model cache: persisted in a Docker volume mounted at `/root/.cache`

## Memory Controls

The service avoids reading full uploads into Python memory and processes long
audio in chunks before handing work to the TensorFlow worker.

- `SEGMENTER_CHUNK_SECONDS`: chunk size for long audio, in seconds. Defaults to `600`.
- `SEGMENTER_CHUNK_MIN_SECONDS`: minimum input duration before chunking is used. Defaults to `900`.
- `SEGMENTER_MERGE_GAP_SECONDS`: max same-label gap to merge after chunk stitching. Defaults to `0.25`.
- `SEGMENTER_UPLOAD_READ_SIZE`: upload streaming buffer size in bytes. Defaults to `1048576`.
- `SEGMENTER_IDLE_TIMEOUT_SECONDS`: warm worker idle lifetime. Defaults to `60`.

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
