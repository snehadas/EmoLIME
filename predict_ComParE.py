
from joblib import load
import numpy as np
import opensmile
import soundfile as sf
import tempfile
import os

# Benchmark against hand-crafted features
smile_ComParE_2016 = opensmile.Smile(
    opensmile.FeatureSet.ComParE_2016,
    opensmile.FeatureLevel.Functionals,
    sampling_rate=16000,
    resample=True,
    num_workers=5,
    verbose=True,
)

model = load("models\EMODB_ComParE_4emo_model.joblib")
# model.classes_ = np.array(['anger', 'happy', 'neutral', 'sad'])

def predict_single(input_audio):
    # Use a temporary file to save the waveform
    
# Create a temporary file
    tmpfile = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        # Write waveform to the temporary WAV file
        sf.write(tmpfile.name, input_audio, 16000)
    finally:
        tmpfile.close()  # Ensure the file is closed

    # Process the audio file with openSMILE
    features = smile_ComParE_2016.process_file(tmpfile.name)

    # Clean up the temporary file
    os.remove(tmpfile.name)

    # predict
    logits = model.predict_proba(features)[0]

    return logits


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