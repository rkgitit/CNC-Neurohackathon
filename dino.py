# train_eeg_classifiers.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---- Load data ----
X = pd.read_csv("eeg_dataset_input.csv")       # Shape: (n_samples, n_channels * window_size)
y = pd.read_csv("eeg_labels.csv.csv")          # Should contain columns: 'label' (either 'nod' or 'blink')

# ---- Feature extraction ----
def extract_features(window: np.ndarray) -> dict:
    feats = {}
    for ch in range(window.shape[1]):
        x = window[:, ch]
        feats[f"ch{ch}_mean"] = float(x.mean())
        feats[f"ch{ch}_std"]  = float(x.std())
        feats[f"ch{ch}_rms"]  = float(np.sqrt(np.mean(x**2)))
    return feats

# ---- Parse windows and labels ----
window_size = 250   # 1 second if 250Hz sampling
n_channels = 8

feature_dicts = []
labels_nod = []
labels_blink = []

for i in range(len(X)):
    row = X.iloc[i].values.reshape((window_size, n_channels))
    feats = extract_features(row)
    feature_dicts.append(feats)

    label = y.iloc[i]['label']
    if label == 'nod':
        labels_nod.append('nod')
        labels_blink.append('other')
    elif label == 'blink':
        labels_nod.append('other')
        labels_blink.append('blink')
    else:
        labels_nod.append('other')
        labels_blink.append('other')

# ---- Vectorize features ----
vec_nod = DictVectorizer(sparse=False)
X_nod = vec_nod.fit_transform(feature_dicts)

vec_blink = DictVectorizer(sparse=False)
X_blink = vec_blink.fit_transform(feature_dicts)

# ---- Train classifiers ----
clf_nod = RandomForestClassifier(n_estimators=100, random_state=42)
clf_nod.fit(X_nod, labels_nod)

clf_blink = RandomForestClassifier(n_estimators=100, random_state=42)
clf_blink.fit(X_blink, labels_blink)

# ---- Evaluate (optional) ----
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_nod, labels_nod, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_blink, labels_blink, test_size=0.2, random_state=42)

print("\n--- Nod Classifier Report ---")
print(classification_report(y_test_n, clf_nod.predict(X_test_n)))

print("\n--- Blink Classifier Report ---")
print(classification_report(y_test_b, clf_blink.predict(X_test_b)))

# ---- Save models and vectorizers ----
joblib.dump(clf_nod, "rf_nod_classifier.joblib")
joblib.dump(clf_blink, "rf_blink_classifier.joblib")
joblib.dump(vec_nod, "vec_nod.joblib")
joblib.dump(vec_blink, "vec_blink.joblib")

print("\nModels and vectorizers saved.")
