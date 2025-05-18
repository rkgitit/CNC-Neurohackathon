from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pyautogui
import numpy as np
import time
from scipy.signal import butter, lfilter
import joblib  # for loading model, scaler, vectorizer

# Load trained model components
model = joblib.load('model.pkl')
vec = joblib.load('vectorizer.pkl')
# scaler = joblib.load('scaler.pkl')  # Optional, if you used one

# === Filter Setup ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

# === Feature Extraction ===
def extract_features(window):
    feats = {}
    for i in range(window.shape[1]):
        ch = window[:, i]
        feats[f'ch{i}_mean'] = np.mean(ch)
        feats[f'ch{i}_std'] = np.std(ch)
        feats[f'ch{i}_rms'] = np.sqrt(np.mean(ch**2))
    return feats

# === Connect to Board ===
params = BrainFlowInputParams()
params.serial_port = "COM4"
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

board.prepare_session()
board.start_stream()

print("Real-time prediction started. Ctrl+C to stop.")

sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
window_sec = 1
num_samples = sampling_rate * window_sec
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

try:
    while True:
        data = board.get_current_board_data(num_samples)
        eeg_data = data[eeg_channels, :].T  # shape: (samples, channels)

        # Preprocess (filter)
        eeg_data = bandpass_filter(eeg_data, fs=sampling_rate)

        # Extract features
        feats = extract_features(eeg_data)
        feats_vec = vec.transform([feats])  # dict to vector

        # If using a scaler
        # feats_vec = scaler.transform(feats_vec)

        prediction = model.predict(feats_vec)[0]
        print("Predicted:", prediction)

        # Map predictions to actions
        if prediction == "blink":
            pyautogui.press("space")
        elif prediction == "nod":
            pyautogui.keyDown("down")
            time.sleep(0.3)
            pyautogui.keyUp("down")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopping...")
    board.stop_stream()
    board.release_session()
