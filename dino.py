import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, lfilter
import pyautogui
import joblib

from dinowork import launch_game  # Your custom game launcher

# ====== Load & Train Model on Stored EEG Data ======
X_df = pd.read_csv("eeg_dataset_input_jaw_up.csv")
y_df = pd.read_csv("eeg_labels_jaw_up.csv")
y = y_df['label'] if 'label' in y_df.columns else y_df.iloc[:, 0]

# def extract_features_from_df(X_array):
#     features_list = []
#     for row in X_array:
#         row_reshaped = row.reshape(-1, 8)  # 8 EEG channels
#         feats = {}
#         for i in range(row_reshaped.shape[1]):
#             ch = row_reshaped[:, i]
#             feats[f'ch{i}_mean'] = np.mean(ch)
#             feats[f'ch{i}_std'] = np.std(ch)
#             feats[f'ch{i}_rms'] = np.sqrt(np.mean(ch**2))
#         features_list.append(feats)
#     return features_list

X_array = X_df.values
# features_dicts = extract_features_from_df(X_array)

vec = joblib.load('vectorizer1.pkl')
# X_vec = vec.transform(features_dicts)                                                               

clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

print("\n--- Classifier Evaluation ---")
print(classification_report(y_test, clf.predict(X_test)))

# clf = joblib.load('mlmodel.pkl')

# ====== Filtering Functions ======
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

# ====== Real-Time Feature Extraction ======
def extract_features(window):
    feats = {}
    for i in range(window.shape[1]):
        ch = window[:, i]
        feats[f'ch{i}_mean'] = np.mean(ch)
        feats[f'ch{i}_std'] = np.std(ch)
        feats[f'ch{i}_rms'] = np.sqrt(np.mean(ch**2))
    return feats

# ====== BrainFlow Setup ======
params = BrainFlowInputParams()
params.serial_port = "COM4"  # ⚠️ Update this to your actual COM port
board = BoardShim(BoardIds.CYTON_BOARD.v    alue, params)

board.prepare_session()
board.start_stream()

# ======    Dino Game Setup ======
print("Launching Chrome  game...")
game = "chrome://dino"
#launch_gam         e("chrome://dino")
launch_game(game)
print("\n🧠 Real-time EEG classification started. Press Ctrl+C to stop.")

sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
window_sec = 1
num_samples = sampling_rate * window_sec
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

last_action_time = 0
cooldown = 0.4
prev_prediction = 'nothing'

try:
    while True:
        raw_data = board.get_current_board_data(num_samples)
        eeg_window = raw_data[eeg_channels, :].T

        # Match preprocessing
        eeg_window = eeg_window - np.mean(eeg_window, axis=0)  # Remove DC
        eeg_window = bandpass_filter(eeg_window, fs=sampling_rate)

        feats = extract_features(eeg_window)
        X_live = vec.transform([feats])
        prediction = clf.predict(X_live)[0]
        # print("Predicted:", prediction)
        # and current_time - last_action_time > cooldown

        current_time = time.time()
        if game == "chrome://dino":
            if (prev_prediction == 'nothing' and prediction == "up_nod"):
                print("JUMP")
                pyautogui.press("space")
                last_action_time = current_time
                # time.sleep(0.1)
            elif prev_prediction == 'nothing' and prediction == "left_nod":
                pyautogui.press("down")
                last_action_time = current_time
                # time.sleep(0.3)
                # pyautogui.keyUp("down")
            prev_prediction = prediction
            # time.sleep(0.1)           
        elif game == "https://www.abcya.com/games/rainbow_stacker":
            # Control Dino Game
            if prediction == "up_nod" and current_time - last_action_time > cooldown:
                pyautogui.click()
                last_action_time = current_time
            elif prediction == "left_nod":
                pyautogui.press("down")
                last_action_time = current_time
            time.sleep(0.1)                                                                                   

except KeyboardInterrupt:
    print("\n🛑 Stopping real-time classification...")
    board.stop_stream()
    board.release_session()