import tensorflow as tf
import numpy as np
import librosa
from preprocessing.detection_preprocess import preprocess_audio_chunk

class CryDetectionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_file(self, audio_path, threshold=0.3, min_cry_seconds=1):
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        cry_counter = 0
        probs = []

        for i in range(0, len(audio) - sr, sr):
            chunk = audio[i:i+sr]

            mel = preprocess_audio_chunk(chunk, sr)
            mel = mel[np.newaxis, ..., np.newaxis]

            prob = float(self.model.predict(mel, verbose=0)[0][0])
            probs.append(prob)

            if prob > threshold:
                cry_counter += 1

        detected = cry_counter >= min_cry_seconds

        return {
            "cry_detected": detected,
            "probabilities_per_second": probs
        }
