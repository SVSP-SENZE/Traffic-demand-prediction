# Flipkart GRiD Hackathon 2.0 — Traffic Demand Prediction

## Problem Statement

Predict normalized traffic demand values for geohash-encoded road segments based on temporal, geographic, road, and weather features. The target variable `demand` is a float in [0, 1] representing relative traffic intensity.

**Metric:** R² Score (higher is better; 100% = perfect fit)

---

## Dataset Overview

| File | Rows | Columns | Description |
|---|---|---|---|
| `train.csv` | 77,299 | 11 | Historical traffic with demand labels |
| `test.csv` | 41,778 | 10 | Unlabeled segments for prediction |
| `sample_submission.csv` | 5 | 2 | Format reference |

**Features:**
- `geohash` — 6-character geohash encoding the road location
- `day` — Day index (48–49 in training data)
- `timestamp` — Time in `H:MM` format (15-min intervals)
- `RoadType` — Type of road (Highway, Residential, etc.)
- `NumberofLanes` — Integer lane count
- `LargeVehicles` — Whether large vehicles are allowed (`Allowed` / `Not Allowed`)
- `Landmarks` — Presence of nearby landmarks (`Yes` / `No`)
- `Temperature` — Ambient temperature in °C (some missing values)
- `Weather` — Weather condition (Sunny, Rainy, Foggy, Snowy; some missing values)
- `demand` *(train only)* — Normalized traffic demand [0, 1]

---

## Approach

### 1. Missing Value Imputation
- `Weather` → mode of training set
- `RoadType` → mode of training set
- `Temperature` → median of training set
- All fill values computed from train only (no data leakage)

### 2. Feature Engineering

**Temporal features from `timestamp`:**
- Extracted `hour` and `minute`
- Cyclical encodings: `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`
- Cyclical day encoding: `day_sin`, `day_cos`
- `is_peak_hour` — flag for 7–9 AM and 5–7 PM
- `is_night` — flag for 12 AM – 6 AM
- `hour_x_day` — interaction term

**Geographic features:**
- Decoded geohash to `lat`, `lon` using `geohash2`
- KMeans geo-clustering: `geo_cluster_4`, `geo_cluster_5` (k=4 and k=5)

**Target encoding (K-Fold, 5 splits, no leakage):**
- `geo_hour_mean_demand` — mean demand per geohash × hour
- `geo_day_mean_demand` — mean demand per geohash × day
- `geo_mean_demand` — mean demand per geohash
- `geo_hour_std_demand` — std demand per geohash × hour
- `geo_day_std_demand` — std demand per geohash × day

**Categorical encoding:**
- `OrdinalEncoder` fitted on combined train + test for `Weather`, `RoadType`, `LargeVehicles`, `Landmarks`
- `LabelEncoder` for `geohash`

### 3. Models

Three gradient boosting models trained with early stopping on a 80/20 train/val split:

| Model | Library | Key Hyperparameters |
|---|---|---|
| LightGBM | `lightgbm` | `n_estimators=2000`, `lr=0.02`, `num_leaves=127` |
| XGBoost | `xgboost` | `n_estimators=2000`, `lr=0.02`, `max_depth=7` |
| CatBoost | `catboost` | `iterations=2000`, `lr=0.02`, `depth=8` |

### 4. Ensemble
- Predictions from all 3 models are blended using **Nelder-Mead weight optimization** on validation R²
- Final predictions are clipped to `[0, ∞)` (demand cannot be negative)

---

## Results

| Stage | RMSE |
|---|---|
| CatBoost Train (final iter) | ~0.0229 |
| CatBoost Val (final iter) | ~0.0280 |

---

## Files Submitted

| File | Description |
|---|---|
| `submission.csv` | Final predictions (41,778 rows, Index + demand) |
| `eda_modeling.ipynb` | Full EDA + feature engineering + modeling notebook |
| `README.md` | This file |
| `comments.txt` | Additional notes and answers |

---

## Dependencies

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
geohash2
scipy
```

Install with:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost geohash2 scipy
```

---

## How to Reproduce

1. Place `train.csv` and `test.csv` in your working directory (update paths in notebook cell 1)
2. Run all cells in `eda_modeling.ipynb` top to bottom
3. `submission.csv` is saved in the same working directory

---

## Team

**Hackathon:** Flipkart GRiD 2.0  
**Problem:** Traffic Demand Prediction
