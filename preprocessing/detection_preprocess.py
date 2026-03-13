import librosa
import numpy as np
import random

def preprocess_audio_chunk(audio, sr=16000):
    # trim silence (زي النوت بوك)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # pre-emphasis (زي النوت بوك)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # normalize (زي النوت بوك)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # fixed length = 1 sec (زي النوت بوك)
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    else:
        audio = audio[:sr]

    # Mel Spectrogram (نفس الإعدادات)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=160
    )

    # Log + Normalize (زي النوت بوك)
    mel = np.log(mel + 1e-9)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel

def predict_file(self, audio_path, threshold=0.3, min_cry_seconds=1):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    probs = []
    cry_counter = 0

    hop = int(sr * 0.5)

    for i in range(0, len(audio) - sr, hop):
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
