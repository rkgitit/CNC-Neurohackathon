# eeg_live_trainer_and_runner.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pyautogui
import time
from scipy.signal import butter, lfilter

# === Load & Prepare Data ===
X_df = pd.read_csv("eeg_dataset_input.csv")         # shape: (samples, ch*window)
y_df = pd.read_csv("eeg_labels.csv.csv")            # single 'label' column
y = y_df['label'] if 'label' in y_df.columns else y_df.iloc[:, 0]

# Feature extraction from training data
def extract_features_from_df(X_array):
    features_list = []
    for row in X_array:
        row_reshaped = row.reshape(-1, 8)  # Assuming 8 channels
        feats = {}
        for i in range(row_reshaped.shape[1]):
            ch = row_reshaped[:, i]
            feats[f'ch{i}_mean'] = np.mean(ch)
            feats[f'ch{i}_std'] = np.std(ch)
            feats[f'ch{i}_rms'] = np.sqrt(np.mean(ch**2))
        features_list.append(feats)
    return features_list

X_array = X_df.values
features_dicts = extract_features_from_df(X_array)

# === Vectorize features ===
vec = DictVectorizer(sparse=False)
X_vec = vec.fit_transform(features_dicts)

# === Train classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

print("\n--- Classifier Evaluation ---")
print(classification_report(y_test, clf.predict(X_test)))

# === Realtime Setup ===

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

def extract_features(window):
    feats = {}
    for i in range(window.shape[1]):
        ch = window[:, i]
        feats[f'ch{i}_mean'] = np.mean(ch)
        feats[f'ch{i}_std'] = np.std(ch)
        feats[f'ch{i}_rms'] = np.sqrt(np.mean(ch**2))
    return feats

# === Connect to EEG board ===
params = BrainFlowInputParams()
params.serial_port = "COM4"  # Update for your setup
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

board.prepare_session()
board.start_stream()

print("\n🧠 Real-time EEG classification started. Press Ctrl+C to stop.")

sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
window_sec = 1
num_samples = sampling_rate * window_sec
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

try:
    while True:
        data = board.get_current_board_data(num_samples)
        eeg_data = data[eeg_channels, :].T  # shape: (samples, channels)

        # Preprocess
        eeg_data = bandpass_filter(eeg_data, fs=sampling_rate)

        # Extract + Vectorize
        feats = extract_features(eeg_data)
        feats_vec = vec.transform([feats])

        # Predict
        prediction = clf.predict(feats_vec)[0]
        print("Predicted:", prediction)

        # Action mapping
        if prediction == "blink":
            pyautogui.press("space")
        elif prediction == "nod":
            pyautogui.keyDown("down")
            time.sleep(0.3)
            pyautogui.keyUp("down")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n🛑 Stopping real-time classification...")
    board.stop_stream()
    board.release_session()
