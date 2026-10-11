import asyncio
import importlib.util
import os
import shutil
import tempfile
from concurrent.futures.process import BrokenProcessPool

import numpy as np

import engine
import main
from main import (
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


def test_blocked_mfcc_matches_upstream():
    spec = importlib.util.spec_from_file_location("upstream_mfcc", engine.sidekit_mfcc.__file__)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    block = engine.SPECTRUM_BLOCK_FRAMES
    for frames in (2, block, block + 1, 3 * block + 1):
        sig = np.random.default_rng(frames).normal(size=(frames - 1) * 160 + 400).astype(np.float32)
        expected, actual = upstream.mfcc(sig, get_mspec=True), engine.sidekit_mfcc.mfcc(sig, get_mspec=True)
        assert np.array_equal(expected[1], actual[1]) and np.array_equal(expected[3], actual[3]), frames


def test_patches_match_upstream():
    for frames in (68, 69, 300, 301):
        mspec = np.random.default_rng(frames).normal(size=(frames, 21)).astype(np.float32)
        mspec[frames // 2] = np.nan
        expected, _ = engine.segmenter._get_patches(mspec, engine.PATCH_FRAMES, engine.PATCH_STEP)
        n_windows = (frames - engine.PATCH_FRAMES) // engine.PATCH_STEP + 1
        actual = engine.patches(mspec, np.arange(len(expected)), n_windows)
        assert np.array_equal(expected.reshape(len(expected), -1), actual, equal_nan=True), frames


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
    test_blocked_mfcc_matches_upstream()
    test_patches_match_upstream()
    test_failed_upload_removes_temp_file()
    print("ok")
