"""Construction feature definitions and preprocessing (UCI Residential Building data).

Features are the 8 project variables V-1..V-8; the target is V-10 (actual construction cost).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import encode_onehot, scale_numeric_features

RAW_PATH = "data/raw/construction_residential.csv"

# Sales price (V-9) is excluded as it is a second outcome, not a pre-construction input; the
# 95 macroeconomic columns are excluded to avoid overfitting 372 rows.
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
ONEHOT_COLS = ["locality"]  # zone code, no inherent order
NUMERIC_COLS = [
    "floor_area_m2", "lot_area_m2", "prelim_est_cost_total",
    "prelim_est_cost", "prelim_est_cost_equiv", "planned_duration",
    "unit_price_start",
]
TARGET_COL = "actual_construction_cost"


def load_construction_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, usecols=list(COLUMN_RENAME))
    return df.rename(columns=COLUMN_RENAME)[list(COLUMN_RENAME.values())]


def preprocess_construction(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder, StandardScaler]:
    """One-hot locality, scale numeric columns, and move the target to the last column."""
    df, encoder = encode_onehot(df, ONEHOT_COLS)
    df, scaler = scale_numeric_features(df, NUMERIC_COLS)
    df[TARGET_COL] = df.pop(TARGET_COL)
    return df, encoder, scaler
