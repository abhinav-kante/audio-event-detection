# 🔊 Live Audio Event Detection for Public Safety

## 📌 Project Overview

This project is an **end-to-end machine learning system** designed to detect and classify **environmental audio events** in real time to support **public safety monitoring**.
It analyzes audio signals, extracts meaningful features, classifies the sound using a trained ML model, and **raises alerts** when potentially dangerous events are detected.

The system focuses on identifying sounds such as **sirens, screams, and explosions**, which are commonly associated with emergency or unsafe situations.

---

## 🎯 Problem Statement

In crowded public places such as railway stations, concerts, campuses, and streets, **critical incidents are often first indicated by sound** rather than visuals.
Manual monitoring is inefficient and error-prone.

This project aims to:

* Automatically analyze audio input
* Identify abnormal or dangerous sounds
* Trigger alerts in real time to enable faster response

---

## 🧠 Solution Approach

The solution is built as a **complete machine learning pipeline**, consisting of:

1. **Audio Preprocessing**
2. **Feature Extraction**
3. **Dataset Construction**
4. **Model Training**
5. **Real-Time Prediction**
6. **Alert Generation**

Each stage is modular, robust, and explainable.

---

## 🔄 System Workflow

1. Audio is captured from a file or live microphone
2. Audio features are extracted using signal processing techniques
3. Extracted features are fed into a trained LightGBM classifier
4. The model predicts the sound category and confidence score
5. Alerts are triggered based on prediction and confidence threshold

---

## 🧩 Key Components Explained

### 1️⃣ Audio Preprocessing

* All audio is standardized to:

  * **WAV format**
  * **22050 Hz sampling rate**
  * **Mono channel**
* This ensures consistency and reliable feature extraction.

---

### 2️⃣ Feature Extraction

Using the **Librosa** library, the following features are extracted:

* **MFCC (Mel-Frequency Cepstral Coefficients)**
  Captures the timbral and perceptual characteristics of sound.

* **Spectral Contrast**
  Measures the difference between spectral peaks and valleys, useful for detecting sharp or harsh sounds.

* **Chroma Features**
  Represents harmonic content of audio.

To handle variable-length audio, **mean and standard deviation** of features are computed, resulting in a fixed-length feature vector (~45 features per audio file).

---

### 3️⃣ Dataset Construction

* Audio files are organized into **class-wise folders**
* Features are extracted from each file
* A structured dataset (`X`, `y`) is created for ML training
* Robust error handling ensures corrupted audio files do not crash the pipeline

---

### 4️⃣ Model Training

* **LightGBM Classifier** is used due to:

  * Fast training
  * High efficiency on structured data
  * Good performance with limited samples
* Labels are encoded using `LabelEncoder`
* Dataset is split into training and testing sets
* Model is evaluated using precision, recall, and F1-score
* Trained model and encoder are saved for reuse

---

### 5️⃣ Real-Time Prediction

The system supports:

* **Audio file prediction**
* **Live microphone input**

For every input:

* Features are extracted
* Model predicts sound category
* Prediction confidence is calculated

---

### 6️⃣ Probability-Based Alert System

To reduce false alarms:

* Alerts are triggered **only if confidence exceeds a threshold**
* Dangerous events include:

  * Siren
  * Scream
  * Explosion

Alert logic:

* **High confidence → Emergency alert**
* **Low confidence → Warning**
* **Normal sound → Safe state**

---

## 🚨 Example Output

```
🎧 Detected Sound: SIREN
📊 Confidence: 0.82
🚨 ALERT: Dangerous event detected with high confidence!
```

---

## 🛠 Tech Stack

* **Programming Language:** Python
* **Audio Processing:** Librosa
* **Machine Learning:** LightGBM, scikit-learn
* **Real-Time Audio:** SoundDevice
* **Model Persistence:** Joblib

---

## 📁 Project Structure

```
p1-audio-event-detection/
├── feature_extraction.py
├── dataset_builder.py
├── train_model.py
├── predict_audio.py
├── live_predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ▶️ How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Build dataset

```bash
python dataset_builder.py
```

### Train model

```bash
python train_model.py
```

### Predict from audio file

```bash
python predict_audio.py
```

### Live microphone detection

```bash
python live_predict.py
```

---

## ⚠️ Limitations

* Model accuracy depends on dataset size and audio quality
* Synthetic or small datasets may result in lower accuracy
* Performance improves significantly with real-world audio samples

---

## 🚀 Future Enhancements

* Integration with CCTV and IoT sensors
* Deployment as a web or mobile application
* Deep learning models (CNNs on spectrograms)
* Cloud-based real-time alerting system

---

## 👤 Author

**Abhinav Kante**
B.Tech – Computer Science & Engineering

---

## 🏆 Why This Project Matters

This project demonstrates:

* Practical application of machine learning
* Real-time audio analysis
* Robust system design
* Industry-relevant problem solving

It is suitable for **resume, GitHub portfolio, interviews, and academic evaluation**.

---

If you want next, I can:

* ✨ Shorten this for **resume**
* ✨ Create **LinkedIn project description**
* ✨ Prepare **interview explanation (1–2 mins)**
* ✨ Start **P2 (NLP project)**

Just tell me 👍
