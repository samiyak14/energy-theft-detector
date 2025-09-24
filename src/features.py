"""
Feature Engineering Script for Energy Theft Detection
File: src/features.py

- Reads output/synthetic_usage.csv and output/house_metadata.csv
- Computes easy-to-understand but useful features per house
- Saves features to output/features.csv
"""

import pandas as pd
import numpy as np
import os

# paths
USAGE_FILE = 'output/synthetic_usage.csv'
META_FILE = 'output/house_metadata.csv'
OUTPUT_FILE = 'output/features.csv'

# create output dir if it doesn't exist
os.makedirs('output', exist_ok=True)

# load data
usage = pd.read_csv(USAGE_FILE)
meta = pd.read_csv(META_FILE)

# prepare feature dataframe
features = meta[['house_id', 'zone_x', 'zone_y', 'label']].copy()

# compute features
feature_list = []
for house_id, group in usage.groupby('house_id'):
    vals = group['consumption_kwh'].values
    mean_c = np.mean(vals)
    std_c = np.std(vals)
    min_c = np.min(vals)
    max_c = np.max(vals)
    pct_zero = np.sum(vals < 0.01) / len(vals) * 100
    max_drop = np.max(np.diff(vals) * -1)  # largest drop
    avg_change = np.mean(np.abs(np.diff(vals)))
    consumption_range = max_c - min_c
    std_ratio = std_c / mean_c if mean_c != 0 else 0

    feature_list.append({
        'house_id': house_id,
        'mean_consumption': mean_c,
        'std_consumption': std_c,
        'min_consumption': min_c,
        'max_consumption': max_c,
        'pct_zero_hours': pct_zero,
        'max_drop': max_drop,
        'avg_hourly_change': avg_change,
        'consumption_range': consumption_range,
        'std_ratio': std_ratio
    })

features_stats = pd.DataFrame(feature_list)

# merge with metadata (zone, label)
features = pd.merge(features[['house_id', 'zone_x', 'zone_y', 'label']], features_stats, on='house_id')

# save features
features.to_csv(OUTPUT_FILE, index=False)
print(f'Features saved to {OUTPUT_FILE}')
print(features.head())
