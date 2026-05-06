import asyncio
import gc
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

IDLE_TIMEOUT_SECONDS = int(os.getenv("SEGMENTER_IDLE_TIMEOUT_SECONDS", "60"))
UPLOAD_READ_SIZE = int(os.getenv("SEGMENTER_UPLOAD_READ_SIZE", str(1024 * 1024)))
CHUNK_SECONDS = int(os.getenv("SEGMENTER_CHUNK_SECONDS", "600"))
CHUNK_MIN_SECONDS = int(os.getenv("SEGMENTER_CHUNK_MIN_SECONDS", "900"))
MERGE_GAP_SECONDS = float(os.getenv("SEGMENTER_MERGE_GAP_SECONDS", "0.25"))

worker_executor: Optional[ProcessPoolExecutor] = None
worker_lock = asyncio.Lock()
worker_unload_task: Optional[asyncio.Task] = None
worker_last_used_at = 0.0
worker_active_requests = 0

_worker_segmenter = None


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def run_segmentation_job(audio_path: str):
    global _worker_segmenter

    if _worker_segmenter is None:
        log.info("Loading INA segmenter model in worker process")
        from inaSpeechSegmenter import Segmenter

        _worker_segmenter = Segmenter(detect_gender=False)
        log.info("INA segmenter model ready in worker process")
    return _worker_segmenter(audio_path)


async def save_upload_to_temp_file(file: UploadFile, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

        while chunk := await file.read(UPLOAD_READ_SIZE):
            tmp.write(chunk)

    return tmp_path


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )


def probe_duration_seconds(audio_path: str) -> Optional[float]:
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("Could not probe audio duration for %s: %s", audio_path, exc)
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        log.warning("ffprobe returned an invalid duration for %s: %r", audio_path, result.stdout)
        return None


def plan_audio_chunks(audio_path: str, duration: float) -> tuple[list[tuple[str, float, Optional[float]]], Optional[str]]:
    if CHUNK_SECONDS <= 0 or duration <= CHUNK_MIN_SECONDS:
        return [(audio_path, 0.0, None)], None

    chunk_dir = tempfile.mkdtemp(prefix="ina-segmenter-chunks-")
    chunks = []
    offset = 0.0
    index = 0

    while offset < duration:
        chunk_path = os.path.join(chunk_dir, f"chunk-{index:04d}.wav")
        chunk_duration = min(float(CHUNK_SECONDS), duration - offset)
        chunks.append((chunk_path, offset, chunk_duration))
        offset += float(CHUNK_SECONDS)
        index += 1

    log.info(
        "Will split %.1fs audio into %d chunks of up to %ss",
        duration,
        len(chunks),
        CHUNK_SECONDS,
    )
    return chunks, chunk_dir


def create_audio_chunk(
    audio_path: str,
    chunk_path: str,
    offset: float,
    chunk_duration: float,
) -> None:
    try:
        run_command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{offset:.3f}",
                "-t",
                f"{chunk_duration:.3f}",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                "-y",
                chunk_path,
            ]
        )
    except Exception:
        if os.path.exists(chunk_path):
            os.unlink(chunk_path)
        raise


def merge_adjacent_segments(segments: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    merged: list[tuple[str, float, float]] = []

    for label, start, end in segments:
        if end <= start:
            continue

        if merged:
            prev_label, prev_start, prev_end = merged[-1]
            if label == prev_label and start - prev_end <= MERGE_GAP_SECONDS:
                merged[-1] = (prev_label, prev_start, max(prev_end, end))
                continue

        merged.append((label, start, end))

    return merged


def get_worker_executor() -> ProcessPoolExecutor:
    global worker_executor

    if worker_executor is None:
        log.info("Starting segmentation worker process")
        worker_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        )
    return worker_executor


def unload_worker() -> None:
    global worker_executor

    if worker_executor is None:
        return

    log.info("Stopping segmentation worker after %ss idle", IDLE_TIMEOUT_SECONDS)
    executor = worker_executor
    worker_executor = None
    executor.shutdown(wait=True, cancel_futures=False)
    gc.collect()


def schedule_unload_locked() -> None:
    global worker_unload_task

    if IDLE_TIMEOUT_SECONDS <= 0:
        return

    if worker_unload_task is not None:
        worker_unload_task.cancel()
    worker_unload_task = asyncio.create_task(unload_when_idle())


async def unload_when_idle() -> None:
    global worker_unload_task

    try:
        await asyncio.sleep(IDLE_TIMEOUT_SECONDS)
        async with worker_lock:
            if worker_active_requests == 0 and worker_executor is not None:
                idle_for = time.monotonic() - worker_last_used_at
                if idle_for >= IDLE_TIMEOUT_SECONDS:
                    await run_in_thread(unload_worker)
    except asyncio.CancelledError:
        pass
    finally:
        if worker_unload_task is asyncio.current_task():
            worker_unload_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_unload_task
    del app

    log.info("INA segmenter worker will start on demand")

    yield

    async with worker_lock:
        if worker_unload_task is not None:
            worker_unload_task.cancel()
            worker_unload_task = None
        await run_in_thread(unload_worker)


app = FastAPI(
    title="INA Speech Segmenter API",
    lifespan=lifespan,
)


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    global worker_active_requests, worker_last_used_at

    async with worker_lock:
        if worker_unload_task is not None:
            worker_unload_task.cancel()
        worker_active_requests += 1

    filename = file.filename or "upload.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"

    log.info("Received file %s", filename)

    tmp_path = None
    chunk_dir = None

    try:
        tmp_path = await save_upload_to_temp_file(file, suffix)
        t0 = time.time()
        loop = asyncio.get_running_loop()
        duration = await run_in_thread(probe_duration_seconds, tmp_path)
        chunks, chunk_dir = await run_in_thread(
            plan_audio_chunks,
            tmp_path,
            duration or 0.0,
        )
        result = []

        async with worker_lock:
            executor = get_worker_executor()

        for index, (chunk_path, offset, chunk_duration) in enumerate(chunks, start=1):
            if len(chunks) > 1:
                log.info(
                    "Segmenting chunk %d/%d for %s at %.1fs",
                    index,
                    len(chunks),
                    filename,
                    offset,
                )

            try:
                if chunk_duration is not None:
                    await run_in_thread(
                        create_audio_chunk,
                        tmp_path,
                        chunk_path,
                        offset,
                        chunk_duration,
                    )

                chunk_result = await loop.run_in_executor(
                    executor,
                    run_segmentation_job,
                    chunk_path,
                )
                result.extend(
                    (label, start + offset, end + offset)
                    for label, start, end in chunk_result
                )
            finally:
                if chunk_duration is not None and os.path.exists(chunk_path):
                    os.unlink(chunk_path)

        result = merge_adjacent_segments(result)
        elapsed = time.time() - t0
        log.info(
            "Segmented %s in %.1fs (%d segments)",
            filename,
            elapsed,
            len(result),
        )
    finally:
        if chunk_dir is not None:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        if tmp_path is not None:
            os.unlink(tmp_path)
        async with worker_lock:
            worker_active_requests -= 1
            worker_last_used_at = time.monotonic()
            schedule_unload_locked()

    return [
        {"label": label, "start": start, "end": end}
        for label, start, end in result
    ]
