"""
EstimateX — Construction domain preprocessing.

Independent pipeline for the UCI Residential Building Data Set
(data/raw/construction_residential.csv). Deliberately NOT forced into the
software domain's schema — this dataset's columns describe a physical
construction project (floor area, lot area, preliminary cost estimates,
planned duration, unit price) with no analog to "team experience" or
COCOMO-style effort-driver ratings, so it gets its own feature set and its
own scaler/encoder, reusing only the generic building blocks from
preprocessing.py (handle_missing_values, encode_onehot,
scale_numeric_features, save_artifact) rather than any COCOMO/Desharnais-
specific logic.

=====================================================================
COLUMN / TARGET DECISIONS — for the paper:
=====================================================================

Source has 109 columns: 4 Persian-calendar date columns, 8 "PROJECT
PHYSICAL AND FINANCIAL VARIABLES" (V-1..V-8), 95 macroeconomic
indicator columns (V-11..V-29, repeated once per each of 5 time lags),
and 2 columns the dataset's own creators label "(output)": V-9 (actual
sales price) and V-10 (actual construction cost).

- Target: V-10, "Actual construction costs (output)" — the direct
  analog to what EstimateX predicts elsewhere ("cost"). V-9 (sales
  price) is a different, though related, quantity (what units sold
  for, not what building them cost) and is DROPPED entirely rather
  than used as a feature — including it as a predictor would be a
  leakage risk, since it's essentially another dataset-provided
  "answer" correlated with construction cost, not a genuine
  pre-construction input.

- Features: only V-1..V-8, the "PROJECT PHYSICAL AND FINANCIAL
  VARIABLES" group — i.e. attributes of the project itself. The 95
  economic-indicator columns (V-11..V-29 x 5 time lags: interest
  rates, price indices, exchange rates, population, gold price, etc.)
  are macroeconomic CONTEXT, not project attributes, and are excluded:
  including 95 mostly-collinear macro columns on only 372 rows would
  risk overfitting and would abandon the project-attribute-level
  explainability framing used everywhere else in EstimateX (COCOMO-NASA
  and Desharnais's features are all about the project, not the
  economy it was built in).

- The 4 raw START/COMPLETION year+quarter columns are also dropped:
  V-7 ("Duration of construction") is already the direct summary of
  that date range, so keeping both would be redundant, and the raw
  Persian-calendar year values are period-specific rather than a
  generalizable numeric signal.

- V-7 (planned/estimated duration) IS kept as an input FEATURE. See
  results/construction_domain_notes.md for why duration is used as a
  predictor here but NOT offered as something this domain predicts —
  the dataset never provides an "actual" duration outcome to train or
  evaluate against, only a pre-construction planned figure.

Zero missing values were found anywhere in the raw sheet (confirmed at
conversion time), so no imputation/row-dropping is needed here — unlike
Desharnais's '?' placeholders in Step 4.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import save_artifact

RAW_PATH = "data/raw/construction_residential.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUT_PATH = f"{PROCESSED_DIR}/construction_clean.csv"

# Raw source column -> clean, descriptive name.
COLUMN_RENAME = {
    "V-1": "locality",
    "V-2": "floor_area_m2",
    "V-3": "lot_area_m2",
    "V-4": "prelim_est_cost_total",
    "V-5": "prelim_est_cost",
    "V-6": "prelim_est_cost_equiv",
    "V-7": "planned_duration",
    "V-8": "unit_price_start",
    "V-10": "actual_construction_cost",
}

# `locality` is a nominal zone/zip-code-style identifier (no inherent
# order among its ~20 values) -> one-hot, not ordinal.
ONEHOT_COLS = ["locality"]
NUMERIC_COLS = [
    "floor_area_m2", "lot_area_m2", "prelim_est_cost_total",
    "prelim_est_cost", "prelim_est_cost_equiv", "planned_duration",
    "unit_price_start",
]
TARGET_COL = "actual_construction_cost"


def load_construction_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, usecols=list(COLUMN_RENAME.keys()))
    df = df.rename(columns=COLUMN_RENAME)
    return df[list(COLUMN_RENAME.values())]  # drop V-9 etc. by not selecting them; fix column order


def preprocess_construction(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder, StandardScaler]:
    df = df.copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[ONEHOT_COLS])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(ONEHOT_COLS), index=df.index)
    df = pd.concat([df.drop(columns=ONEHOT_COLS), encoded_df], axis=1)

    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    target = df.pop(TARGET_COL)
    df[TARGET_COL] = target
    return df, encoder, scaler


if __name__ == "__main__":
    print("EstimateX — construction domain preprocessing")
    print()

    raw_df = load_construction_raw()
    print(f"Raw shape (features + target, V-9/economic columns/dates excluded): {raw_df.shape}")
    print(f"Missing values: {int(raw_df.isnull().sum().sum())}")
    print()

    processed, encoder, scaler = preprocess_construction(raw_df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}  shape={processed.shape}")

    save_artifact(scaler, f"{MODELS_DIR}/construction_scaler.joblib")
    save_artifact(encoder, f"{MODELS_DIR}/construction_onehot_encoder.joblib")
    print(f"Saved -> {MODELS_DIR}/construction_scaler.joblib")
    print(f"Saved -> {MODELS_DIR}/construction_onehot_encoder.joblib")
