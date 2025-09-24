"""
Isolation Forest Training + Scoring for Energy Theft Detection
File: src/train_isolation_forest.py

- Reads output/features.csv
- Trains Isolation Forest (unsupervised anomaly detection)
- Saves model to models/isolation_forest.pkl
- Outputs predictions CSV: output/predictions.csv with house_id and risk_score
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import joblib

# paths
FEATURES_FILE = 'output/features.csv'
MODEL_FILE = 'models/isolation_forest.pkl'
PRED_FILE = 'output/predictions.csv'

# create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# load features
features = pd.read_csv(FEATURES_FILE)
metadata_cols = ['house_id', 'zone_x', 'zone_y', 'label']
feature_cols = [c for c in features.columns if c not in metadata_cols]
X = features[feature_cols]

# train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# anomaly score (the lower, the more abnormal)
scores = -model.decision_function(X)  # invert to make higher = more anomalous

# save model
joblib.dump(model, MODEL_FILE)
print(f'Model saved to {MODEL_FILE}')

# prepare predictions dataframe
predictions = pd.DataFrame({
    'house_id': features['house_id'],
    'risk_score': scores
})

# save predictions
predictions.to_csv(PRED_FILE, index=False)
print(f'Predictions saved to {PRED_FILE}')
print(predictions.head())
