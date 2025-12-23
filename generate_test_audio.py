import numpy as np
import soundfile as sf

# Generate 2 seconds of dummy audio (sine wave)
sr = 22050
t = np.linspace(0, 2, int(sr * 2))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

sf.write("data/siren/siren.wav", audio, sr)

print("Valid siren.wav created")
