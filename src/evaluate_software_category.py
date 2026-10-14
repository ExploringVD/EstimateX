"""
EstimateX — software "Project category" accuracy evaluation.

app.py derives Project category from the deployed software model's
predicted effort alone (120/190 person-month thresholds around the 150
PM reference point — see app.py's PROJECT_CATEGORY_* constants) — no
separate classifier is trained. This script checks how often that
derived category matches reality, using the SAME out-of-fold CV
predictions a fair evaluation requires (never the in-sample fit).

Deliberately named "category accuracy," never "model accuracy" — this
measures a binary threshold applied AFTER regression, not a
classification model's own performance, and the two are kept visibly
distinct in every print statement below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from train_software_cocomo import (
    CV_FOLDS,
    CV_REPEATS,
    FULL_COLUMNS,
    RAW_COCOMO_PATH,
    COCOMO_TARGET_COL,
    build_cocomo_features,
    build_cocomo_model,
)

CATEGORY_THRESHOLD_PM = 150.0  # must match app.py's category reference point


def main() -> None:
    df = pd.read_csv(RAW_COCOMO_PATH)
    X = build_cocomo_features(df)[FULL_COLUMNS]
    y_raw = df[COCOMO_TARGET_COL]
    y_log = np.log(y_raw)

    actual_above = (y_raw >= CATEGORY_THRESHOLD_PM).to_numpy()
    majority_baseline = max(actual_above.mean(), 1 - actual_above.mean()) * 100.0

    accuracies = []
    borderline_counts = []
    for seed in CV_REPEATS:
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        pred_log = cross_val_predict(build_cocomo_model(), X, y_log, cv=kfold)
        pred_raw = np.exp(pred_log)

        pred_above = pred_raw >= CATEGORY_THRESHOLD_PM
        accuracies.append(float((pred_above == actual_above).mean() * 100.0))
        borderline_counts.append(int(((pred_raw >= 120) & (pred_raw < 190)).sum()))

    accuracies = np.array(accuracies)
    borderline_counts = np.array(borderline_counts)

    print("=" * 78)
    print("EstimateX — Project category accuracy (software, COCOMO-NASA)")
    print("=" * 78)
    print(f"Threshold: predicted/actual effort >= {CATEGORY_THRESHOLD_PM:g} PM")
    print(f"Majority-class baseline (always predict the larger class): {majority_baseline:.2f}%")
    print(
        f"Project category accuracy, out-of-fold {CV_FOLDS}-fold CV x {len(CV_REPEATS)} repeats "
        f"(seeds {CV_REPEATS[0]}-{CV_REPEATS[-1]}): mean={accuracies.mean():.2f}%  std={accuracies.std():.2f}%"
    )
    print(
        f"Borderline band (120-190 PM predicted) row count per repeat: "
        f"mean={borderline_counts.mean():.1f}  min={borderline_counts.min()}  max={borderline_counts.max()} "
        f"(out of {len(y_raw)} rows)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
