"""
Synthetic Electricity Usage Data Generator for AI-Based Energy Theft Detection
File: energy_theft_data_generator.py

What it does:
- Generates hourly smart-meter-like data for multiple houses over a date range.
- Simulates normal consumption with daily + weekly patterns.
- Injects several theft/anomaly types: sudden drops (tapping), flatlines (stuck meter), spikes (tampering), and gradual baseline reduction (bypass).
- Assigns each house to a zone with (x,y) coordinates for heatmap visualization.
- Saves CSV: `synthetic_usage.csv` and metadata `house_metadata.csv`.

Usage (example):
    python energy_theft_data_generator.py --n_houses 80 --days 90 --seed 42

Dependencies: pandas, numpy

"""

import argparse
from datetime import timedelta
import numpy as np
import pandas as pd
import os


def generate_house_base_profile(index, periods, freq='H', seed=0):
    """Create a realistic base consumption time series for one house.
    Uses a daily sinusoid + weekday/weekend multiplier + random noise.
    Returns a numpy array of length `periods` with kWh values.
    """
    rng = np.random.default_rng(seed + index)
    t = np.arange(periods)

    # Daily cycle: sinusoid with 24-hour period (if freq is hourly)
    daily = 1.5 + 1.0 * np.sin(2 * np.pi * (t % 24) / 24 - 0.5)

    # Weekday multiplier: slightly higher usage weekdays
    weekday = 1.0 + 0.15 * ((t // 24) % 7 < 5)

    # House-specific base load and variability
    base = 0.2 + rng.normal(0.0, 0.05) + 0.05 * (index % 5)
    variability = 0.05 + 0.02 * rng.random()

    noise = rng.normal(0, variability, size=periods)

    profile = (base + daily) * weekday + noise
    profile = np.clip(profile, 0.01, None)
    return profile


def inject_anomalies(profile, rng, types_allowed=None):
    """Inject anomalies into a copy of profile. Returns profile_copy, list_of_events.
    types_allowed is a list of anomaly types to choose from.
    """
    p = profile.copy()
    events = []
    if types_allowed is None:
        types_allowed = ['sudden_drop', 'flatline', 'spikes', 'gradual_drop']

    # Decide randomly whether this house is anomalous
    if rng.random() > 0.25:  # 75% normal by default
        return p, events

    # Choose 1-2 anomaly events
    n_events = rng.integers(1, 3)
    periods = len(p)

    for _ in range(n_events):
        a_type = rng.choice(types_allowed)
        start = int(rng.integers(24, max(48, periods - 48)))
        duration = int(rng.integers(6, min(240, periods - start)))

        if a_type == 'sudden_drop':
            # sudden drop during peak hours (reduce by 50-90%)
            reduction = rng.uniform(0.5, 0.9)
            p[start:start + duration] *= (1 - reduction)
            events.append((a_type, start, duration, {'reduction': reduction}))

        elif a_type == 'flatline':
            # meter stuck: almost constant low reading
            level = rng.uniform(0.01, 0.2) * np.nanmean(profile)
            p[start:start + duration] = level
            events.append((a_type, start, duration, {'level': float(level)}))

        elif a_type == 'spikes':
            # random spikes: tampering to show high sudden readings
            n_spikes = max(1, duration // 24)
            for i in range(n_spikes):
                pos = start + int(rng.integers(0, max(1, duration)))
                if pos < periods:
                    p[pos:pos + 1] += rng.uniform(3, 8)  # big spike
            events.append((a_type, start, duration, {'n_spikes': n_spikes}))

        elif a_type == 'gradual_drop':
            # baseline slowly reduced (bypass)
            reduction = rng.uniform(0.2, 0.6)
            end = min(periods, start + duration)
            ramp = np.linspace(1.0, 1.0 - reduction, end - start)
            p[start:end] *= ramp
            events.append((a_type, start, duration, {'reduction': reduction}))

    p = np.clip(p, 0.001, None)
    return p, events


def generate_dataset(n_houses=50, days=60, freq='H', seed=0, out_dir='output'):
    rng = np.random.default_rng(seed)
    periods = days * 24  # hourly data

    os.makedirs(out_dir, exist_ok=True)

    timestamps = pd.date_range('2023-01-01', periods=periods, freq=freq)

    rows = []
    meta = []

    for h in range(n_houses):
        house_id = f'H_{h:03d}'
        # assign zones (grid neighborhoods)
        zone_x = float(rng.integers(0, 10)) + rng.random()
        zone_y = float(rng.integers(0, 10)) + rng.random()

        base_profile = generate_house_base_profile(h, periods, seed=seed)
        profile, events = inject_anomalies(base_profile, rng)

        # Save metadata
        label = 'normal'
        if len(events) > 0:
            # classify by severity: any sudden_drop or flatline -> highly suspicious
            severity = 0
            for e in events:
                if e[0] in ('flatline', 'sudden_drop'):
                    severity += 2
                elif e[0] in ('gradual_drop', 'spikes'):
                    severity += 1
            if severity >= 2:
                label = 'highly_suspicious'
            else:
                label = 'suspicious'

        meta.append({
            'house_id': house_id,
            'zone_x': zone_x,
            'zone_y': zone_y,
            'n_events': len(events),
            'label': label,
            'events': str(events)
        })

        for ts_idx, ts in enumerate(timestamps):
            rows.append({
                'house_id': house_id,
                'timestamp': ts,
                'consumption_kwh': float(profile[ts_idx])
            })

    df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)

    # Save CSVs
    out_data = os.path.join(out_dir, 'synthetic_usage.csv')
    out_meta = os.path.join(out_dir, 'house_metadata.csv')
    df.to_csv(out_data, index=False)
    meta_df.to_csv(out_meta, index=False)

    print(f'Generated dataset for {n_houses} houses, {days} days (periods={periods}).')
    print(f'Data saved to: {out_data}')
    print(f'Metadata saved to: {out_meta}')

    return df, meta_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic smart-meter electricity usage data')
    parser.add_argument('--n_houses', type=int, default=50)
    parser.add_argument('--days', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='output')

    args = parser.parse_args()
    generate_dataset(n_houses=args.n_houses, days=args.days, seed=args.seed, out_dir=args.out_dir)
