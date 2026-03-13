from utils.yamnet_utils import extract_yamnet_embedding
import numpy as np

def preprocess_for_analysis(audio_path):
    emb = extract_yamnet_embedding(audio_path)
    return emb.reshape(1, -1)   # (1, 3072)
