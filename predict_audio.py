import joblib
from feature_extraction import extract_features

# Load trained model and label encoder
model = joblib.load("lgbm_audio_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Dangerous sound classes
DANGEROUS_EVENTS = ["siren", "scream", "explosion"]
CONFIDENCE_THRESHOLD = 0.6


def predict_audio(file_path):
    features = extract_features(file_path)

    if features is None:
        print("⚠️ Could not process audio")
        return

    features = features.reshape(1, -1)

    # Prediction and probability
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = encoder.inverse_transform([prediction])[0]
    confidence = max(probabilities)

    # Output
    print(f"🎧 Detected Sound: {label.upper()}")
    print(f"📊 Confidence: {confidence:.2f}")

    if label in DANGEROUS_EVENTS and confidence >= CONFIDENCE_THRESHOLD:
        print("🚨 ALERT: Dangerous event detected with high confidence!")
    elif label in DANGEROUS_EVENTS:
        print("⚠️ Possible dangerous sound (low confidence)")
    else:
        print("✅ Normal environment")


# Example usage
if __name__ == "__main__":
    test_audio = "data/siren/siren_1.wav"  # change path to test other sounds
    predict_audio(test_audio)
