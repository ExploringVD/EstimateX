"""Read-only accuracy check for both deployed models; nothing is saved.

Software uses 10x repeated 10-fold CV (no test set at 93 rows); construction uses its held-out
test set. Run from the project root: python check_accuracy.py
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import train_software_cocomo as sw  # noqa: E402

CATEGORY_THRESHOLD_PM = 150  # must match app.py's PROJECT_CATEGORY_REFERENCE_PM
BORDERLINE_BAND_PM = (120, 190)  # must match app.py's borderline band


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


df = pd.read_csv(sw.RAW_COCOMO_PATH)
X = sw.build_cocomo_features(df)[sw.FULL_COLUMNS]
y_raw = df[sw.COCOMO_TARGET_COL]

line("SOFTWARE — COCOMO sign-constrained Ridge (10x repeated 10-fold CV)")
cv = sw.repeated_cv_metrics(X, y_raw)
for col in sw.METRIC_COLUMNS:
    print(f"  {col:9s}: {cv[col].mean():8.3f}  (+/- {cv[col].std():.3f})")
print("  -> PRED(25) = % of projects predicted within ±25% of actual effort")

line("CONSTRUCTION — deployed Random Forest on held-out test set")
m = joblib.load("models/final_model_construction.pkl")
te = pd.read_csv("data/processed/construction_test.csv")
y = te["actual_construction_cost"].values
p = m.predict(te[m.feature_names_in_])
mre = np.abs(y - p) / y
print(f"  test rows : {len(y)}")
print(f"  R2        : {r2_score(y, p):.3f}")
print(f"  PRED(25)  : {(mre <= .25).mean() * 100:.1f}%")
print(f"  MAE       : {mean_absolute_error(y, p):.2f}  (x10,000 IRR)")
print(f"  MMRE      : {mre.mean():.3f}")
print(f"  MdMRE     : {np.median(mre):.3f}")

# The app's project category is a threshold on predicted effort, so it is scored on the same
# out-of-fold predictions, never on the in-sample fit.
line(f"SOFTWARE — project category accuracy (>= {CATEGORY_THRESHOLD_PM} PM, out-of-fold CV)")
actual_above = (y_raw >= CATEGORY_THRESHOLD_PM).to_numpy()
accuracies, borderline = [], []
for seed in sw.CV_REPEATS:
    pred = np.exp(sw.oof_predictions(X, np.log(y_raw), seed))
    accuracies.append(((pred >= CATEGORY_THRESHOLD_PM) == actual_above).mean() * 100)
    borderline.append(((pred >= BORDERLINE_BAND_PM[0]) & (pred < BORDERLINE_BAND_PM[1])).sum())
print(f"  accuracy  : {np.mean(accuracies):.2f}%  (+/- {np.std(accuracies):.2f})")
print(f"  baseline  : {max(actual_above.mean(), 1 - actual_above.mean()) * 100:.2f}%  (always predict the larger class)")
print(f"  borderline: {np.mean(borderline):.1f} of {len(y_raw)} projects per repeat fall in "
      f"{BORDERLINE_BAND_PM[0]}-{BORDERLINE_BAND_PM[1]} PM")
