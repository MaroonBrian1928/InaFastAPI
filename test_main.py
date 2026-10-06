import asyncio
import os
import shutil
from concurrent.futures.process import BrokenProcessPool

import main
from main import merge_adjacent_segments, plan_audio_chunks, run_in_worker


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


if __name__ == "__main__":
    test_plan_audio_chunks()
    test_merge_adjacent_segments()
    test_run_in_worker_replaces_dead_worker()
    print("ok")
