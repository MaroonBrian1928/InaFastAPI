# INA Speech Segmenter API

A lightweight FastAPI wrapper around
[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
that accepts audio file uploads and returns labeled time segments
(speech, music, noise, etc.). The service is designed to run in Docker on
CPU via ONNX Runtime. Local development uses `mise` for tool management and
`uv` for Python dependency management and command execution.

## Service

- Container: `inafastapi-ina-segmenter-1`
- Host port: `8002`
- Container port: `8000`
- Endpoint: `POST /segment`
- Model: loaded on demand in a dedicated worker process with `detect_gender=False`
- Long uploads: split into bounded audio chunks before segmentation to bound peak memory
- Worker lifecycle: kept warm while requests are active, then terminated after the idle timeout to release memory back to the OS
- Model: INA's speech/music/noise CNN exported to ONNX at `models/speech_music_noise_cnn.onnx` (`scripts/export_onnx.py`); `engine.py` runs INA's pipeline on it without TensorFlow

## Memory Controls

The service avoids reading full uploads into Python memory and processes long
audio in chunks before handing work to the segmentation worker.

- `SEGMENTER_CHUNK_SECONDS`: chunk size for long audio, in seconds. Defaults to `600`. Changing it changes results slightly, since INA sets its silence threshold per chunk.
- `SEGMENTER_CHUNK_MIN_SECONDS`: minimum input duration before chunking is used. Defaults to `900`.
- `SEGMENTER_TAIL_MERGE_SECONDS`: a final chunk shorter than this joins the previous chunk. Defaults to `60`.
- `SEGMENTER_MERGE_GAP_SECONDS`: max same-label gap to merge after chunk stitching. Defaults to `0.25`.
- `SEGMENTER_UPLOAD_READ_SIZE`: upload streaming buffer size in bytes. Defaults to `1048576`.
- `SEGMENTER_IDLE_TIMEOUT_SECONDS`: warm worker idle lifetime. Defaults to `60`.
- `SEGMENTER_THREADS`: ONNX Runtime intra-op threads. Defaults to `0` (one per physical core). Set it to the container's CPU limit, which ONNX Runtime cannot see.

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

## Parity

`engine.py` must produce the same segments as upstream inaSpeechSegmenter.
After upgrading inaSpeechSegmenter, ONNX Runtime, or numpy, run `scripts/parity.py`
against a set of real recordings; its docstring has the commands.

## Segment Labels

INA commonly emits labels such as `speech`, `music`, `noise`, and
`noEnergy`, identifying the dominant audio content in each detected
time window.
