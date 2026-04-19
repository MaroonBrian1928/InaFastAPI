import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from inaSpeechSegmenter import Segmenter

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

segmenter: Segmenter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global segmenter

    log.info("Loading INA segmenter model")
    segmenter = Segmenter(detect_gender=False)
    log.info("INA segmenter model ready")
    yield


app = FastAPI(
    title="INA Speech Segmenter API",
    lifespan=lifespan,
)


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Segmenter not ready")

    filename = file.filename or "upload.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"

    log.info("Received file %s", filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        result = segmenter(tmp_path)
        elapsed = time.time() - t0
        log.info(
            "Segmented %s in %.1fs (%d segments)",
            filename,
            elapsed,
            len(result),
        )
    finally:
        os.unlink(tmp_path)

    return [
        {"label": label, "start": start, "end": end}
        for label, start, end in result
    ]
