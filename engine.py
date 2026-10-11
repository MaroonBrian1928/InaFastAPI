"""inaSpeechSegmenter 0.8.0's Segmenter(detect_gender=False) pipeline, run on ONNX Runtime with bounded memory.

Output must stay identical to the upstream TensorFlow pipeline; scripts/parity.py checks that on real audio.
Import this only in the worker process: it replaces the tensorflow and keras modules for the whole process.
"""

import os
import sys
import types

import numpy as np

# inaSpeechSegmenter imports TensorFlow and Keras at module load only to fetch and load its Keras model.
for _name in ("tensorflow", "tensorflow.keras", "tensorflow.keras.utils", "keras", "keras.utils"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["tensorflow"].keras = sys.modules["tensorflow.keras"]
sys.modules["tensorflow.keras"].utils = sys.modules["tensorflow.keras.utils"]
sys.modules["keras"].utils = sys.modules["keras.utils"]
sys.modules["tensorflow.keras.utils"].get_file = sys.modules["keras.utils"].get_file = None

import onnxruntime as ort
from inaSpeechSegmenter import segmenter, sidekit_mfcc
from inaSpeechSegmenter.pyannote_viterbi import viterbi_decoding
from inaSpeechSegmenter.viterbi_utils import diag_trans_exp

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "speech_music_noise_cnn.onnx")
THREADS = int(os.getenv("SEGMENTER_THREADS", "0"))

VAD = segmenter.SpeechMusicNoise
ENERGY_RATIO = 0.03
PATCH_FRAMES = 68
PATCH_STEP = 2
PATCH_BATCH = 32
SPECTRUM_BLOCK_FRAMES = 2000


def power_spectrum(input_sig, fs=8000, win_time=0.025, shift=0.01, prefac=0.97):
    """sidekit_mfcc.power_spectrum with small frame blocks; upstream processes 500000 frames per block."""
    window_length = round(win_time * fs)
    framed = sidekit_mfcc.framing(input_sig, window_length, win_shift=int(shift * fs))
    n_frames = framed.shape[0]
    n_fft = 1 << (window_length - 1).bit_length()
    window = np.hanning(window_length)
    spec = np.ones((n_frames, n_fft // 2 + 1), dtype=sidekit_mfcc.PARAM_TYPE)
    log_energy = np.empty(n_frames, dtype=input_sig.dtype)
    for start in range(0, n_frames, SPECTRUM_BLOCK_FRAMES):
        block = sidekit_mfcc.pre_emphasis(framed[start:start + SPECTRUM_BLOCK_FRAMES].copy(), prefac)
        log_energy[start:start + SPECTRUM_BLOCK_FRAMES] = np.log((block**2).sum(axis=1))
        mag = np.fft.rfft(block * window, n_fft, axis=-1)
        spec[start:start + SPECTRUM_BLOCK_FRAMES] = mag.real**2 + mag.imag**2
    return spec, log_energy


sidekit_mfcc.power_spectrum = power_spectrum

_options = ort.SessionOptions()
_options.intra_op_num_threads = THREADS
_session = ort.InferenceSession(MODEL_PATH, _options, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def patches(mspec, indices, n_windows):
    """Rows `indices` of segmenter._get_patches(mspec, PATCH_FRAMES, PATCH_STEP), flattened."""
    windows = np.clip(indices - PATCH_FRAMES // (2 * PATCH_STEP), 0, n_windows - 1)
    data = np.stack([mspec[PATCH_STEP * w:PATCH_STEP * w + PATCH_FRAMES].reshape(-1) for w in windows])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (data - data.mean(axis=1).reshape(-1, 1)) / data.std(axis=1).reshape(-1, 1)


def classify(mspec, start, stop, n_windows):
    preds = []
    for batch_start in range(start, stop, PATCH_BATCH):
        data = patches(mspec, np.arange(batch_start, min(batch_start + PATCH_BATCH, stop)), n_windows)
        # The model's first layer reshapes to (68, 21, 1), so any input with the same element order works.
        pred = _session.run(None, {_input_name: data.reshape(-1, 1, PATCH_FRAMES, VAD.nmel).astype(np.float32)})[0]
        pred[~np.all(np.isfinite(data), axis=1)] = 0.5
        preds.append(pred)
    path = viterbi_decoding(np.log(np.concatenate(preds)), diag_trans_exp(VAD.viterbi_arg, len(VAD.outlabels)))
    return [(VAD.outlabels[int(label)], s + start, e + start) for label, s, e in segmenter._binidx2seglist(path)]


def segment(audio_path: str) -> list[tuple[str, float, float]]:
    mspec, loge, _ = segmenter._media2feats(audio_path, None, None, "ffmpeg")
    mspec = np.ascontiguousarray(mspec[:, :VAD.nmel])
    n_windows = (len(mspec) - PATCH_FRAMES) // PATCH_STEP + 1
    result = []
    energy = segmenter._energy_activity(loge, ENERGY_RATIO)[::2]
    for active, start, stop in segmenter._binidx2seglist(energy):
        if active:
            result.extend(classify(mspec, start, stop, n_windows))
        else:
            result.append(("noEnergy", start, stop))
    return [(label, start * 0.02, stop * 0.02) for label, start, stop in result]
