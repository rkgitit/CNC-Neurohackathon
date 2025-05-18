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



# ---- Train classifiers ----
clf_nod = RandomForestClassifier(n_estimators=100, random_state=42)
clf_nod.fit(x, y)

clf_blink = RandomForestClassifier(n_estimators=100, random_state=42)
clf_blink.fit(x, y)

# ---- Evaluate (optional) ----
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(x, y, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n--- Nod Classifier Report ---")
print(classification_report(y_test_n, clf_nod.predict(X_test_n)))

print("\n--- Blink Classifier Report ---")
print(classification_report(y_test_b, clf_blink.predict(X_test_b)))

# ---- Save models and vectorizers ----
joblib.dump(clf_nod, "rf_nod_classifier.joblib")
joblib.dump(clf_blink, "rf_blink_classifier.joblib")
joblib.dump(y, "vec_nod.joblib")
joblib.dump(y, "vec_blink.joblib")

print("\nModels and vectorizers saved.")
