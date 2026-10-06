import asyncio
import os
import shutil
import tempfile
from concurrent.futures.process import BrokenProcessPool

import main
from main import (
    discard_corrupt_model,
    merge_adjacent_segments,
    plan_audio_chunks,
    run_in_worker,
    save_upload_to_temp_file,
)


def chunk_spans(duration):
    chunks, chunk_dir = plan_audio_chunks("in.mp3", duration)
    if chunk_dir:
        shutil.rmtree(chunk_dir)
    return [(offset, length) for _, offset, length in chunks]


def test_plan_audio_chunks():
    assert chunk_spans(900) == [(0.0, None)]
    assert chunk_spans(1500) == [(0.0, 600.0), (600.0, 600.0), (1200.0, 300.0)]
    assert chunk_spans(1200.03) == [(0.0, 600.0), (600.0, 600.03)]
    assert chunk_spans(1300) == [(0.0, 600.0), (600.0, 600.0), (1200.0, 100.0)]


def test_merge_adjacent_segments():
    assert merge_adjacent_segments(
        [("speech", 0, 600), ("speech", 600.1, 700), ("music", 700, 710), ("noise", 5, 5)]
    ) == [("speech", 0, 700), ("music", 700, 710)]


def exit_worker():
    os._exit(1)


def test_run_in_worker_replaces_dead_worker():
    async def scenario():
        try:
            await run_in_worker(exit_worker)
        except BrokenProcessPool:
            pass
        else:
            raise AssertionError("expected BrokenProcessPool")
        assert main.worker_executor is None
        assert await run_in_worker(os.getpid) != os.getpid()
        main.unload_worker()

    asyncio.run(scenario())


def test_discard_corrupt_model():
    with tempfile.TemporaryDirectory() as tmp:
        main.MODEL_PATH = os.path.join(tmp, "model.hdf5")
        discard_corrupt_model()
        with open(main.MODEL_PATH, "wb") as f:
            f.write(b"truncated")
        discard_corrupt_model()
        assert not os.path.exists(main.MODEL_PATH)


class FailingUpload:
    async def read(self, size):
        raise OSError("connection reset")


def test_failed_upload_removes_temp_file():
    before = set(os.listdir(tempfile.gettempdir()))
    try:
        asyncio.run(save_upload_to_temp_file(FailingUpload(), ".mp3"))
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError")
    assert set(os.listdir(tempfile.gettempdir())) == before


if __name__ == "__main__":
    test_plan_audio_chunks()
    test_merge_adjacent_segments()
    test_run_in_worker_replaces_dead_worker()
    test_discard_corrupt_model()
    test_failed_upload_removes_temp_file()
    print("ok")
