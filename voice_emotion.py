import librosa
import numpy as np
import joblib
MODEL = joblib.load("voice_emotion_model.pkl")
def detect_voice_emotion(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    pred = MODEL.predict([mfcc])[0]
    prob = max(MODEL.predict_proba([mfcc])[0])
    return pred, float(prob)