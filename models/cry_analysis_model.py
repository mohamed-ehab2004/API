import tensorflow as tf
import numpy as np
from preprocessing.analysis_preprocess import preprocess_for_analysis

CLASS_NAMES = ["hunger", "sleepy", "discomfort", "pain", "fear"]

class CryAnalysisModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, audio_path):
        features = preprocess_for_analysis(audio_path)
        probs = self.model.predict(features)[0]
        pred_class = int(np.argmax(probs))
        return CLASS_NAMES[pred_class], probs.tolist()
