
from joblib import load
import numpy as np
import torchaudio
import soundfile as sf
import tempfile
import os
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch

# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

clf = load("../models/EMODB_w2v2 FT dim_4emo_model.joblib")
# clf.classes_ = np.array(['anger', 'happy', 'neutral', 'sad']) 

def predict_single(input_audio):
    # check if waveform is mono or stereo and convert to mono by averaging channels. wav2vec2 expects a single-channel (mono) input
    if input_audio.ndim > 1:
        input_audio = input_audio.mean(axis=1)

    input = processor(input_audio.squeeze(), sampling_rate=16000, return_tensors="pt")#.input_values

    with torch.no_grad():
        outputs = model(**input)

    # .mean(axis = 0) applies average pooling to reduce dimensionality
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis = 0).numpy()

    logits = clf.predict_proba(last_hidden_states.reshape(1, -1))

    return logits[0]


def predict(input_audio):
    print("Prediction function called")
    if isinstance(input_audio, list) or len(np.shape(input_audio)) > 1:

        # various files, need loop
        batch_size = len(input_audio)
        scores = np.zeros((batch_size, 1))
        for i, audio in enumerate(input_audio):
            scores[i, 0] = predict_single(audio)[0] # angry

    else:
        # just predict on single file
        scores = predict_single(input_audio)[0] # angry

    print(f"Prediction score: {scores}")
    return scores