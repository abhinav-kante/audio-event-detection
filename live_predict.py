import sounddevice as sd
import numpy as np
import librosa
import joblib
from feature_extraction import extract_features
import soundfile as sf

model = joblib.load("lgbm_audio_model.pkl")
encoder = joblib.load("label_encoder.pkl")

DANGEROUS_EVENTS = ["siren", "scream", "explosion"]
CONFIDENCE_THRESHOLD = 0.6
SAMPLE_RATE = 22050
DURATION = 3  # seconds

def record_audio():
    print("🎙 Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write("temp.wav", audio, SAMPLE_RATE)
    print("✅ Recording complete")

def predict_live():
    record_audio()
    features = extract_features("temp.wav")

    if features is None:
        print("⚠️ Audio could not be processed")
        return

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    label = encoder.inverse_transform([prediction])[0]

    print(f"🎧 Detected: {label.upper()} | Confidence: {confidence:.2f}")

    if label in DANGEROUS_EVENTS and confidence >= CONFIDENCE_THRESHOLD:
        print("🚨 LIVE ALERT!")
    else:
        print("✅ Safe")

if __name__ == "__main__":
    predict_live()
