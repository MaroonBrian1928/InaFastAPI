import asyncio
import gc
import logging
import multiprocessing as mp
import os
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
        executor = get_worker_executor()
        worker_active_requests += 1

    filename = file.filename or "upload.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"

    log.info("Received file %s", filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_segmentation_job, tmp_path)
        elapsed = time.time() - t0
        log.info(
            "Segmented %s in %.1fs (%d segments)",
            filename,
            elapsed,
            len(result),
        )
    finally:
        os.unlink(tmp_path)
        async with worker_lock:
            worker_active_requests -= 1
            worker_last_used_at = time.monotonic()
            schedule_unload_locked()

    return [
        {"label": label, "start": start, "end": end}
        for label, start, end in result
    ]
