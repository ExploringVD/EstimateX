"""
EstimateX — Step 5: train/test splitting.

This module is the SINGLE SOURCE OF TRUTH for how every dataset gets split
into train/test. Step 6 will compare four models (Linear Regression,
Random Forest, XGBoost, SVR) per dataset — they must all be evaluated on
the exact same rows, so the split parameters live here as constants, not
as magic numbers scattered across training scripts, and every split in
this project should go through `split_dataframe()` below rather than
calling `train_test_split` directly elsewhere.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"

# Fixed once, reused everywhere: Step 6 must see the same train/test rows
# for every model it compares, or the comparison between models wouldn't
# be fair.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# One entry per cleaned dataset produced by src/preprocessing.py.
DATASET_CONFIGS = [
    {
        "name": "COCOMO-NASA",
        "clean_path": f"{PROCESSED_DIR}/cocomo_nasa_clean.csv",
        "target_col": "act_effort",
        "train_path": f"{PROCESSED_DIR}/cocomo_nasa_train.csv",
        "test_path": f"{PROCESSED_DIR}/cocomo_nasa_test.csv",
    },
    {
        "name": "Desharnais",
        "clean_path": f"{PROCESSED_DIR}/desharnais_clean.csv",
        "target_col": "Effort",
        "train_path": f"{PROCESSED_DIR}/desharnais_train.csv",
        "test_path": f"{PROCESSED_DIR}/desharnais_test.csv",
    },
]


def split_features_target(df: pd.DataFrame, target_col: str):
    """Separate a dataframe into (X, y): every column except `target_col`
    as the feature matrix, `target_col` itself as the target series.

    Shared by every place in the project that needs to pull features and
    target apart from an already-assembled CSV (the initial split below,
    and later reloading of the saved train/test files in
    src/train_models.py and src/evaluate.py) — kept as one function so
    that logic isn't copy-pasted across files.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_dataframe(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split a cleaned dataframe into X_train, X_test, y_train, y_test.

    `target_col` is dropped from the feature matrix and returned
    separately as y — this is the one place in the project that should
    ever call `train_test_split`, so every model in Step 6 is trained and
    evaluated on an identical split.
    """
    X, y = split_features_target(df, target_col)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str,
    train_path: str,
    test_path: str,
) -> None:
    """Write train/test features + target back out as two combined CSVs
    (same "features then target as the last column" layout that
    src/preprocessing.py already uses for the cleaned datasets), so Step 6
    can just `pd.read_csv(...)` and split off `target_col` again without
    re-running `train_test_split`.
    """
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df = X_train.copy()
    train_df[target_col] = y_train
    train_df.to_csv(train_path, index=False)

    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv(test_path, index=False)


def split_and_save_dataset(
    name: str,
    clean_path: str,
    target_col: str,
    train_path: str,
    test_path: str,
) -> dict:
    """Full pipeline for one dataset: load the cleaned CSV, split it, save
    the train/test CSVs, and return a summary of the resulting shapes.
    """
    df = pd.read_csv(clean_path)
    X_train, X_test, y_train, y_test = split_dataframe(df, target_col)
    save_split(X_train, X_test, y_train, y_test, target_col, train_path, test_path)

    return {
        "name": name,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "train_path": train_path,
        "test_path": test_path,
    }


def _print_summary(summary: dict) -> None:
    print("=" * 70)
    print(summary["name"])
    print("-" * 70)
    print(f"X_train: {summary['X_train_shape']}   y_train: {summary['y_train_shape']}")
    print(f"X_test:  {summary['X_test_shape']}   y_test:  {summary['y_test_shape']}")
    print(f"Saved -> {summary['train_path']}, {summary['test_path']}")
    print()


if __name__ == "__main__":
    print("EstimateX — Step 5 train/test split summary")
    print(f"(random_state={RANDOM_STATE}, test_size={TEST_SIZE})")
    print()
    for cfg in DATASET_CONFIGS:
        result = split_and_save_dataset(**cfg)
        _print_summary(result)
