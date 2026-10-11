"""Check that engine.py matches upstream inaSpeechSegmenter on real audio, through the service's chunking.

Run the reference with TensorFlow, the candidate without it, then compare:
  uv run --with tensorflow-cpu==2.21.0 python scripts/parity.py run reference OUT_REF FILELIST
  uv run python scripts/parity.py run engine OUT_NEW FILELIST
  uv run python scripts/parity.py compare OUT_REF OUT_NEW

FILELIST has one audio path per line. Outputs are one JSON file per input; existing ones are skipped.
"""

import json
import os
import resource
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

LABELS = ["speech", "music", "noise", "noEnergy"]


def segment_file(segment, path):
    duration = main.probe_duration_seconds(path)
    chunks, chunk_dir = main.plan_audio_chunks(path, duration or 0.0)
    result = []
    try:
        for chunk_path, offset, chunk_duration in chunks:
            if chunk_duration is not None:
                main.create_audio_chunk(path, chunk_path, offset, chunk_duration)
            result.extend((label, start + offset, end + offset) for label, start, end in segment(chunk_path))
    finally:
        if chunk_dir is not None:
            shutil.rmtree(chunk_dir, ignore_errors=True)
    return duration, main.merge_adjacent_segments(result)


def run(engine_name, out_dir, filelist):
    if engine_name == "reference":
        from inaSpeechSegmenter import Segmenter

        segment = Segmenter(detect_gender=False)
    else:
        from engine import segment
    os.makedirs(out_dir, exist_ok=True)
    with open(filelist) as f:
        paths = f.read().splitlines()
    for path in paths:
        out = os.path.join(out_dir, os.path.basename(path) + ".json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            duration, segments = segment_file(segment, path)
            record = {"duration": duration, "elapsed": time.time() - t0, "segments": segments}
        except Exception as exc:
            record = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        with open(out, "w") as f:
            json.dump(record, f)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10
        print(f"{time.time() - t0:7.1f}s  peak {peak} MiB  {record.get('error', 'ok')[:60]}  {os.path.basename(path)}", flush=True)


def frames(segments, n):
    import numpy as np

    labels = np.full(n, -1, np.int8)
    for label, start, end in segments:
        labels[round(start * 100):round(end * 100)] = LABELS.index(label)
    return labels


def load(path):
    with open(path) as f:
        return json.load(f)


def compare(ref_dir, new_dir):
    files = identical = 0
    agree = total = 0
    ref_time = new_time = 0.0
    for name in sorted(os.listdir(ref_dir)):
        files += 1
        new_path = os.path.join(new_dir, name)
        if not os.path.exists(new_path):
            print(f"MISSING  {name}")
            continue
        ref, new = load(os.path.join(ref_dir, name)), load(new_path)
        if "error" in ref or "error" in new:
            print(f"ERROR    {name}: reference={ref.get('error', 'ok')!r} engine={new.get('error', 'ok')!r}")
            continue
        ref_time += ref["elapsed"]
        new_time += new["elapsed"]
        if [tuple(s) for s in ref["segments"]] == [tuple(s) for s in new["segments"]]:
            identical += 1
        else:
            print(f"DIFFERS  {name}")
        n = round(max(ref["segments"][-1][2], new["segments"][-1][2]) * 100)
        a, b = frames(ref["segments"], n), frames(new["segments"], n)
        agree += int((a == b).sum())
        total += n
    print(f"{identical}/{files} identical, {total / 360000:.1f} h audio, "
          f"10 ms label agreement {100 * agree / max(total, 1):.3f}%, "
          f"time {ref_time:.0f}s -> {new_time:.0f}s")
    return identical == files


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:5])
    else:
        sys.exit(0 if compare(*sys.argv[2:4]) else 1)
