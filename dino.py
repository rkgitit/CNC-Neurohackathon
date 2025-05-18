# dino_eeg_control.py
import time
import joblib
import numpy as np
from pyautogui import press
from scipy.signal import butter, lfilter
import dinowork
import pandas as pd
import datajump


df = pd.read_csv("eeg_dataset_input.csv")
df2 = pd.read_csv("eeg_labels.csv.csv")


# —— Configuration ——
GAMELINK = "chrome://dino"
MODEL_NOD_PATH   = "rf_nod_classifier.joblib"
MODEL_BLINK_PATH = "rf_blink_classifier.joblib"
VEC_NOD_PATH     = "vec_nod.joblib"
VEC_BLINK_PATH   = "vec_blink.joblib"

# EEG params
FS = 250
WINDOW_SEC = 1
WINDOW_SIZE = int(FS * WINDOW_SEC)
LOWCUT, HIGHCUT = 0.5, 40.0

# Load ML models
clf_nod   = joblib.load(MODEL_NOD_PATH)
clf_blink = joblib.load(MODEL_BLINK_PATH)
vec_nod   = joblib.load(VEC_NOD_PATH)
vec_blink = joblib.load(VEC_BLINK_PATH)

# Signal filters

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

# Feature extraction

def extract_features(window: np.ndarray) -> dict:
    feats = {}
    for ch in range(window.shape[1]):
        x = window[:, ch]
        feats[f"ch{ch}_mean"] = float(x.mean())
        feats[f"ch{ch}_std"]  = float(x.std())
        feats[f"ch{ch}_rms"]  = float(np.sqrt(np.mean(x**2)))
    return feats

# Main
if __name__ == "__main__":
    # Launch game
    dinowork.launch_game(GAMELINK)
    time.sleep(1)

    while True:
        # Get raw EEG window
        eeg_window = datajump.get_window(duration=WINDOW_SEC, fs=FS)
        # Filter
        filtered = bandpass_filter(eeg_window)
        # Extract features
        feats = extract_features(filtered)

        # Nod classification
        x_nod = vec_nod.transform(feats)
        pred_nod = clf_nod.predict(x_nod)[0]
        if pred_nod == 'nod':
            print("Nod → Jump")
            press('up')
            time.sleep(0.2)
            continue

        # Blink classification
        x_blink = vec_blink.transform(feats)
        pred_blink = clf_blink.predict(x_blink)[0]
        if pred_blink == 'blink':
            print("Blink → Duck")
            press('down')
            time.sleep(0.2)
            continue

        # No event
        time.sleep(0.05)
