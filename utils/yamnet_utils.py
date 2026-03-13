import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def load_audio_for_yamnet(path, sr=16000, max_sec=4):
    audio, _ = librosa.load(path, sr=sr, mono=True)

    max_len = sr * max_sec
    if len(audio) > max_len:
        audio = audio[:max_len]

    if len(audio) < max_len:
        pad_len = max_len - len(audio)
        audio = np.pad(audio, (0, pad_len))

    # Normalization
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio.astype(np.float32)

def extract_yamnet_embedding(path):
    audio = load_audio_for_yamnet(path)

    scores, embeddings, _ = yamnet(audio)
    # embeddings: (frames, 1024)

    mean = tf.reduce_mean(embeddings, axis=0)
    std  = tf.math.reduce_std(embeddings, axis=0)
    mx   = tf.reduce_max(embeddings, axis=0)

    emb = tf.concat([mean, std, mx], axis=0)  # (3072,)
    return np.array(emb)
    return emb.numpy()
