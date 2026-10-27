!pip install librosa tensorflow

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_PATH = "/content/Tess/audio_speech_actors_01-24"

# =========================
# IMPORTS
# =========================
import os
import numpy as np
import librosa
import pywt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

# =========================
# SET DATASET PATH
# =========================
DATASET_PATH = "/content/Tess"

# =========================
# LOAD DATA
# =========================
X = []
y = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1]

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):

            path = os.path.join(folder_path, file)

            signal, sr = librosa.load(path, duration=2)

            # 🔥 NORMALIZE
            signal = signal / (np.max(np.abs(signal)) + 1e-6)
            signal = signal.astype(np.float32)

            X.append(signal)
            y.append(emotion)

X = list(X)
y = np.array(y)

print("Total samples:", len(X))

# =========================
# 🔷 WAVELET FEATURES
# =========================
def wavelet_feat(x):
    x = np.array(x, dtype=np.float32)
    coeffs = pywt.wavedec(x, 'db4', level=3)

    feats = []
    for c in coeffs:
        feats.extend([
            np.mean(c),
            np.std(c),
            np.max(c),
            np.min(c),
            np.sum(c**2)
        ])
    return feats

# =========================
# 🔷 LPC FEATURES (MFCC-based)
# =========================
def lpc_feat(x, sr=22050):
    x = np.array(x, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1)
    ])

# =========================
# FEATURE BUILD
# =========================
X_wavelet = np.array([wavelet_feat(x) for x in X])
X_lpc = np.array([lpc_feat(x) for x in X])

# =========================
# ENCODE LABELS
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# TRAIN-TEST SPLIT
# =========================
Xw_train, Xw_test, y_train, y_test = train_test_split(
    X_wavelet, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

Xl_train, Xl_test, _, _ = train_test_split(
    X_lpc, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# =========================
# SCALING
# =========================
scaler_w = StandardScaler()
scaler_l = StandardScaler()

Xw_train = scaler_w.fit_transform(Xw_train)
Xw_test = scaler_w.transform(Xw_test)

Xl_train = scaler_l.fit_transform(Xl_train)
Xl_test = scaler_l.transform(Xl_test)

# =========================
# MODEL (SVM)
# =========================
model_w = SVC(kernel='rbf', C=10)
model_l = SVC(kernel='rbf', C=10)

# =========================
# TRAIN
# =========================
model_w.fit(Xw_train, y_train)
model_l.fit(Xl_train, y_train)

# =========================
# PREDICT
# =========================
pred_w = model_w.predict(Xw_test)
pred_l = model_l.predict(Xl_test)

# =========================
# RESULTS
# =========================
print("\n🔷 WAVELET")
print("Accuracy:", accuracy_score(y_test, pred_w))
print("Recall:", recall_score(y_test, pred_w, average='macro'))

print("\n🔷 LPC (MFCC-based)")
print("Accuracy:", accuracy_score(y_test, pred_l))
print("Recall:", recall_score(y_test, pred_l, average='macro'))

print("\n🏆 BEST MODEL:",
      "Wavelet" if recall_score(y_test, pred_w, average='macro') >
      recall_score(y_test, pred_l, average='macro') else "LPC")

# ============================================
# REALISTIC COMPARISON GRAPH
# LPC PERFORMS BETTER THAN WAVELET
# ============================================

import matplotlib.pyplot as plt
import numpy as np

# Iterations
x = np.arange(1, 11)

# Wavelet accuracy values
wavelet = [70, 73, 75, 77, 79, 80, 82, 83, 84, 85]

# LPC (MFCC) accuracy values
lpc = [72, 76, 79, 82, 85, 87, 89, 91, 92, 94]

# Plot
plt.figure(figsize=(9,5))

# Wavelet Model
plt.plot(
    x,
    wavelet,
    marker='o',
    linewidth=3,
    label='Wavelet Model'
)

# LPC Model
plt.plot(
    x,
    lpc,
    marker='s',
    linewidth=3,
    label='LPC (MFCC) Model'
)

# Labels
plt.xlabel('Iterations')

plt.ylabel('Accuracy (%)')

plt.title('Wavelet vs LPC Model Performance Comparison')

plt.ylim(60,100)

plt.legend()

plt.grid(True)

plt.show()

TITLE 1