"""
EstimateX — READ-ONLY accuracy check for both deployed domains.

Nothing here saves, retrains-to-disk, or overwrites any model/data file.
Run from the project root:   python check_accuracy.py

- Software (COCOMO-NASA, 93 projects): there is no separate test set, so it
  uses the same 10x repeated 10-fold cross-validation as
  src/train_software_cocomo.py (models are refit in memory only).
- Construction: the deployed model is scored on its held-out test set
  (data/processed/construction_test.csv), which it never saw in training.
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

import train_software_cocomo as sw  # noqa: E402  (only imported, its main() is NOT run)


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------- SOFTWARE ----------------
line("SOFTWARE — COCOMO sign-constrained Ridge (10x repeated 10-fold CV)")
df = pd.read_csv(sw.RAW_COCOMO_PATH)
X = sw.build_cocomo_features(df)[sw.FULL_COLUMNS]
y_raw = df[sw.COCOMO_TARGET_COL]
cv = sw.repeated_cv_metrics(sw.build_cocomo_model, X, y_raw, np.log(y_raw), predict_is_log=True)
for col in sw.METRIC_COLUMNS:
    print(f"  {col:9s}: {cv[col].mean():8.3f}  (+/- {cv[col].std():.3f})")
print("  -> PRED(25) = % of projects predicted within ±25% of actual effort")

# ---------------- CONSTRUCTION ----------------
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
print("\nFor software 'project category accuracy' run: python src/evaluate_software_category.py")
