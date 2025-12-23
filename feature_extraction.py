import numpy as np
import librosa

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, mono=True)

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_mean = np.mean(spectral_contrast, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)

        return np.hstack([mfcc_mean, mfcc_std, spectral_mean, chroma_mean])

    except Exception as e:
        print(f"❌ Skipping file {file_path} | Error: {e}")
        return None
