"""Regenerate models/speech_music_noise_cnn.onnx from inaSpeechSegmenter's Keras model.

  uv run --with tensorflow-cpu==2.21.0 --with tf2onnx python scripts/export_onnx.py
"""

import os

import numpy as np
from inaSpeechSegmenter.remote_utils import get_remote
from tensorflow import keras

model = keras.models.load_model(get_remote("keras_speech_music_noise_cnn.hdf5"), compile=False)
model(np.zeros((1, 68, 21, 1), np.float32))
model.export(os.path.join(os.path.dirname(__file__), "..", "models", "speech_music_noise_cnn.onnx"), format="onnx")
