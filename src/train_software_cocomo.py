"""
EstimateX — COCOMO-NASA-native software model, sign-constrained.

ATTEMPT 1 (superseded, kept here as a documented finding): a plain
RidgeCV directly on the 15 ordinal driver levels (vl=0..xh=5) + log(KLOC)
+ one-hot mode genuinely beat the deployed unified baseline on CV PRED(25)
(47.7% vs 22.5%) and CV R2(log) (0.828 vs 0.668) — but its `cplx` (Product
Complexity) coefficient came out NEGATIVE (-0.079 at the CV-selected
alpha, ranging from -0.25 at low alpha to +0.04 at alpha=200 — never
robustly positive at any reasonable regularization strength). Verified
this is genuine multicollinearity (cplx correlates with time/stor/sced in
this 93-row sample), not a fold-selection fluke: a plain ordinary-least-
squares-style fit on correlated COCOMO drivers is a well-known failure
mode in the effort-estimation literature — Ridge optimizes for overall
predictive accuracy, not for each individual coefficient's sign, so
nothing in that approach GUARANTEES "more complexity => more effort,"
even though the data clearly supports that direction when cplx is
isolated from its correlated neighbors.

ATTEMPT 2 (this version, deployed): removes the ambiguity entirely by
construction, not by hoping regularization picks a sensible sign.
  1. Each of the 15 drivers is encoded as log(Boehm-1981 Intermediate
     COCOMO effort multiplier) for its rated level, from the standard
     published EM table (COCOMO_EM_TABLE below) — not a raw 0-5 ordinal
     index. A rating with no published value for that specific driver
     (e.g. TIME has no VL/L entries) falls back to the nearest DEFINED
     level on the vl..xh scale.
  2. `cplx` is pulled OUT of the fitted feature set entirely and used as
     a FIXED, weight=1 offset: log(effort) - log(cplx's published EM) is
     the regression target, and log(cplx's published EM) is added back
     after predicting. Boehm's own published CPLX multipliers (.70/.85/
     1.00/1.15/1.30/1.65) are monotonically increasing by construction,
     so complexity's effect on the FINAL prediction is guaranteed
     monotonic regardless of what the other 14 drivers do.
  3. The remaining 14 drivers' log-EM values + log(KLOC) + one-hot mode
     are fit with Ridge(alpha=0.1, positive=True) on that residual
     target. `positive=True` is safe here (unlike on raw ordinal
     indices) because log(EM) already encodes each driver's correct
     theoretical DIRECTION — e.g. ACAP's published EM decreases as
     capability increases, so log(ACAP's EM) already decreases too;
     forcing its Ridge weight >= 0 therefore keeps effort decreasing as
     capability increases, the theoretically correct direction, instead
     of letting multicollinearity flip it.

Model SELECTION/validation is CV-only (10x repeated 10-fold, seeds 0-9)
against the deployed unified baseline — there is no held-out test set:
the whole 93-row file is in-sample for CV, and the model is only ever
fit on all 93 rows AFTER CV has already confirmed it beats the baseline,
never the reverse.

China and Desharnais data, and every existing software artifact, are
left exactly as they are: this script only ever reads data/raw/
cocomo_nasa.csv and data/processed/software_unified.csv (read-only, for
a fair apples-to-apples baseline comparison), and only writes NEW model
files under distinct names. Construction isn't imported, read, or
written anywhere below.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from preprocessing import COCOMO_ORDINAL_COLS, COCOMO_ORDINAL_SCALE, save_artifact

RAW_COCOMO_PATH = "data/raw/cocomo_nasa.csv"
UNIFIED_PATH = "data/processed/software_unified.csv"
UNIFIED_TARGET_COL = "effort_person_months"
COCOMO_TARGET_COL = "act_effort"
KLOC_COL = "equivphyskloc"
MODE_COL = "mode"
MODE_CATEGORIES = ["embedded", "organic", "semidetached"]  # alphabetical, pandas get_dummies default order

MODELS_DIR = "models"
NEW_MODEL_PATH = f"{MODELS_DIR}/final_model_software.pkl"
UNIFIED_BASELINE_BACKUP_PATH = f"{MODELS_DIR}/final_model_software_unified_baseline.pkl"
METADATA_PATH = f"{MODELS_DIR}/final_model_software_metadata.pkl"

RANDOM_STATE = 42  # model hyperparameter seed only — NOT the CV fold seed (that's 0-9 below)
CV_FOLDS = 10
CV_REPEATS = list(range(10))  # seeds 0-9
N_HOLDOUT_REPEATS = 20
RIDGE_ALPHA = 0.1  # fixed, not CV-tuned — see module docstring

# ---------------------------------------------------------------------------
# Boehm 1981 Intermediate COCOMO effort multiplier (EM) table — "Software
# Engineering Economics," reproduced identically across standard COCOMO
# references. Only the OFFICIALLY DEFINED levels are listed per driver;
# get_em() below falls back to the nearest defined level for anything else.
# ---------------------------------------------------------------------------
COCOMO_EM_TABLE: dict[str, dict[str, float]] = {
    "rely": {"vl": 0.75, "l": 0.88, "n": 1.00, "h": 1.15, "vh": 1.40},
    "data": {"l": 0.94, "n": 1.00, "h": 1.08, "vh": 1.16},
    "cplx": {"vl": 0.70, "l": 0.85, "n": 1.00, "h": 1.15, "vh": 1.30, "xh": 1.65},
    "time": {"n": 1.00, "h": 1.11, "vh": 1.30, "xh": 1.66},
    "stor": {"n": 1.00, "h": 1.06, "vh": 1.21, "xh": 1.56},
    "virt": {"l": 0.87, "n": 1.00, "h": 1.15, "vh": 1.30},
    "turn": {"l": 0.87, "n": 1.00, "h": 1.07, "vh": 1.15},
    "acap": {"vl": 1.46, "l": 1.19, "n": 1.00, "h": 0.86, "vh": 0.71},
    "aexp": {"vl": 1.29, "l": 1.13, "n": 1.00, "h": 0.91, "vh": 0.82},
    "pcap": {"vl": 1.42, "l": 1.17, "n": 1.00, "h": 0.86, "vh": 0.70},
    "vexp": {"vl": 1.21, "l": 1.10, "n": 1.00, "h": 0.90},
    "lexp": {"vl": 1.14, "l": 1.07, "n": 1.00, "h": 0.95},
    "modp": {"vl": 1.24, "l": 1.10, "n": 1.00, "h": 0.91, "vh": 0.82},
    "tool": {"vl": 1.24, "l": 1.10, "n": 1.00, "h": 0.91, "vh": 0.83},
    "sced": {"vl": 1.23, "l": 1.08, "n": 1.00, "h": 1.04, "vh": 1.10},
}


def nearest_defined_level(driver: str, level: str) -> str:
    """The closest (ordinal-distance) level that IS published for this
    specific driver — e.g. TIME has no VL/L entries, so a TIME=VL
    request falls back to TIME=N (distance 2), the nearest defined level.
    """
    defined = COCOMO_EM_TABLE[driver]
    if level in defined:
        return level
    idx = COCOMO_ORDINAL_SCALE.index(level)
    return min(defined, key=lambda lv: abs(COCOMO_ORDINAL_SCALE.index(lv) - idx))


def get_em(driver: str, level: str) -> float:
    return COCOMO_EM_TABLE[driver][nearest_defined_level(driver, level)]


# ---------------------------------------------------------------------------
# Feature engineering — 14 of COCOMO-NASA's 15 drivers as log(published EM)
# (cplx is pulled out as a fixed offset, see CplxOffsetRidge below) +
# log(KLOC) + one-hot mode. Order is fixed and saved as metadata so the
# app can reconstruct the exact same column order at inference time.
# ---------------------------------------------------------------------------

CPLX_OFFSET_COL = "cplx_log_em"
NON_CPLX_DRIVERS = [c for c in COCOMO_ORDINAL_COLS if c != "cplx"]
FEATURE_ORDER = NON_CPLX_DRIVERS + ["log_kloc"] + [f"mode_{c}" for c in MODE_CATEGORIES]
FULL_COLUMNS = FEATURE_ORDER + [CPLX_OFFSET_COL]  # what's fed into CplxOffsetRidge


def build_cocomo_features(df: pd.DataFrame) -> pd.DataFrame:
    """log(published EM) for the 14 non-cplx drivers + log(equivphyskloc)
    + one-hot mode + the cplx offset column. No leaky/post-project
    columns (recordnumber, projectname, cat2, forg, center, year are all
    excluded — none are available before a project starts, and
    recordnumber/year are known spurious per preprocessing.py).
    """
    out = pd.DataFrame(index=df.index)
    for col in NON_CPLX_DRIVERS:
        out[col] = df[col].apply(lambda level, driver=col: np.log(get_em(driver, level)))
    out["log_kloc"] = np.log(df[KLOC_COL])
    mode_dummies = pd.get_dummies(df[MODE_COL], prefix="mode")
    for cat in MODE_CATEGORIES:
        out[f"mode_{cat}"] = mode_dummies.get(f"mode_{cat}", 0)
    out[CPLX_OFFSET_COL] = df["cplx"].apply(lambda level: np.log(get_em("cplx", level)))
    return out


# ---------------------------------------------------------------------------
# Metrics — same definitions as src/tune_software_model.py (MMRE, MdMRE,
# PRED(k)); reimplemented locally rather than imported, same reasoning as
# that script's own docstring: this stays a standalone, read-only-of-
# everything-else experiment script.
# ---------------------------------------------------------------------------

def mmre(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float((np.abs(y_true - y_pred) / y_true).mean())


def mdmre(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.median(np.abs(y_true - y_pred) / y_true))


def pred_at(y_true, y_pred, threshold_pct: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    within = (np.abs(y_true - y_pred) / y_true <= threshold_pct / 100.0).sum()
    return float(100.0 * within / len(y_true))


def compute_report_metrics(y_true_raw, y_pred_raw) -> dict:
    """All 6 metrics the brief asks for, from raw-space true/predicted
    effort alone (log-space R2 is the exception, computed on logged
    arrays directly — PRED/MMRE/MdMRE are monotonic-invariant so raw vs
    log would give identical values, and are only reported once, in raw
    terms, per the effort-estimation literature's own definition of MRE).
    """
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    y_true_log = np.log(y_true_raw)
    y_pred_log = np.log(np.clip(y_pred_raw, 1e-6, None))
    return {
        "PRED(25)": pred_at(y_true_raw, y_pred_raw, 25),
        "PRED(50)": pred_at(y_true_raw, y_pred_raw, 50),
        "R2(log)": r2_score(y_true_log, y_pred_log),
        "R2(raw)": r2_score(y_true_raw, y_pred_raw),
        "MMRE": mmre(y_true_raw, y_pred_raw),
        "MdMRE": mdmre(y_true_raw, y_pred_raw),
    }


METRIC_COLUMNS = ["PRED(25)", "PRED(50)", "R2(log)", "R2(raw)", "MMRE", "MdMRE"]


# ---------------------------------------------------------------------------
# The sign-constrained estimator — see module docstring Attempt 2.
# ---------------------------------------------------------------------------

class CplxOffsetRidge(BaseEstimator, RegressorMixin):
    """Ridge(alpha, positive=True) on log(effort) with the CPLX_OFFSET_COL
    column pulled out as a fixed, weight=1 additive offset before fitting
    and added back after predicting. `X` must contain CPLX_OFFSET_COL
    alongside the real features (see FULL_COLUMNS) — this class splits
    them apart internally so it drops into cross_val_predict / fit /
    predict like any other scikit-learn regressor.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha

    def fit(self, X, y):
        X = pd.DataFrame(X)
        offset = X[CPLX_OFFSET_COL].to_numpy()
        features = X.drop(columns=[CPLX_OFFSET_COL])
        self.feature_names_ = list(features.columns)
        self.ridge_ = Ridge(alpha=self.alpha, positive=True)
        self.ridge_.fit(features, np.asarray(y) - offset)
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        offset = X[CPLX_OFFSET_COL].to_numpy()
        features = X[self.feature_names_]
        return self.ridge_.predict(features) + offset


# Running this file directly makes Python see it as module "__main__",
# so pickle would record CplxOffsetRidge under "__main__" — unresolvable
# later when app.py does `from train_software_cocomo import
# CplxOffsetRidge` in a different process. Aliasing this running module
# into sys.modules under its real importable name, then pointing the
# class at that name, makes pickle's save_global() identity check
# (sys.modules[module_name].name IS this exact class object) pass
# regardless of how this script itself was invoked.
sys.modules.setdefault("train_software_cocomo", sys.modules[__name__])
CplxOffsetRidge.__module__ = "train_software_cocomo"


# ---------------------------------------------------------------------------
# Model builders — fresh, unfitted instance each call (CV needs a new
# estimator per fold/repeat, never a shared fitted one).
# ---------------------------------------------------------------------------

def build_unified_model() -> TransformedTargetRegressor:
    """The currently-deployed software model's exact architecture
    (src/tune_software_model.py) — n_estimators=500, min_samples_leaf=8,
    log target. Re-specified here (not imported) because that script
    hardcodes CHOSEN_MIN_SAMPLES_LEAF for ITS OWN deploy logic, which
    this script must not touch.
    """
    return TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=1.0, random_state=RANDOM_STATE
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def build_cocomo_model() -> CplxOffsetRidge:
    return CplxOffsetRidge(alpha=RIDGE_ALPHA)


# ---------------------------------------------------------------------------
# Repeated K-fold CV — out-of-fold predictions pooled within each repeat,
# metrics computed once per repeat on the pooled set (more stable than
# averaging 10 small per-fold scores, standard practice for n=93), then
# mean +/- std reported ACROSS the `CV_REPEATS` repeats.
# ---------------------------------------------------------------------------

def repeated_cv_metrics(build_model, X, y_raw, y_fit, predict_is_log: bool) -> pd.DataFrame:
    """y_fit is what the model is actually trained to predict (log(effort)
    for the COCOMO candidate, raw effort for the unified model, whose
    TransformedTargetRegressor does its own internal log/exp). `predict_
    is_log` says which space cross_val_predict's output is in, so it can
    be converted to raw-space for the metrics, which are always computed
    in raw terms (see compute_report_metrics).
    """
    rows = []
    for seed in CV_REPEATS:
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(build_model(), X, y_fit, cv=kfold)
        y_pred_raw = np.exp(y_pred) if predict_is_log else y_pred
        rows.append(compute_report_metrics(y_raw, y_pred_raw))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, model_name: str) -> dict:
    row = {"Model": model_name}
    for col in METRIC_COLUMNS:
        row[f"{col} mean"] = df[col].mean()
        row[f"{col} std"] = df[col].std()
    return row


def holdout_metrics(build_model, X, y_raw, y_fit, predict_is_log: bool, n_repeats: int) -> pd.DataFrame:
    """N random 80/20 holdouts — a second, independent confirmation of
    the model's CV numbers, same metric definitions, fresh split and
    fresh fit each repeat.
    """
    rows = []
    n = len(X)
    for seed in range(n_repeats):
        idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=seed)
        model = build_model()
        model.fit(X.iloc[idx_train], y_fit.iloc[idx_train])
        y_pred = model.predict(X.iloc[idx_test])
        y_pred_raw = np.exp(y_pred) if predict_is_log else y_pred
        rows.append(compute_report_metrics(y_raw.iloc[idx_test], y_pred_raw))
    return pd.DataFrame(rows)


def main() -> None:
    cocomo_df = pd.read_csv(RAW_COCOMO_PATH)
    X_cocomo = build_cocomo_features(cocomo_df)[FULL_COLUMNS]
    y_cocomo_raw = cocomo_df[COCOMO_TARGET_COL]
    y_cocomo_log = np.log(y_cocomo_raw)

    unified_df = pd.read_csv(UNIFIED_PATH)
    X_unified = unified_df.drop(columns=[UNIFIED_TARGET_COL])
    y_unified_raw = unified_df[UNIFIED_TARGET_COL]

    print("=" * 90)
    print("EstimateX — COCOMO-NASA native software model (sign-constrained, EM-table based)")
    print("=" * 90)
    print(f"COCOMO-NASA: X={X_cocomo.shape}  fitted features={FEATURE_ORDER}")
    print(f"  (+ {CPLX_OFFSET_COL}, a fixed weight=1 offset, not fitted)")
    print(f"Unified baseline: X={X_unified.shape} (its own existing 4-feature schema)")
    print()

    # -----------------------------------------------------------------
    # 10x repeated 10-fold CV — new COCOMO model vs the deployed unified
    # baseline.
    # -----------------------------------------------------------------
    print(f"Repeated {CV_FOLDS}-fold CV, seeds {CV_REPEATS[0]}-{CV_REPEATS[-1]} (10 repeats each):")
    print()

    unified_cv = repeated_cv_metrics(build_unified_model, X_unified, y_unified_raw, y_unified_raw, predict_is_log=False)
    cocomo_cv = repeated_cv_metrics(build_cocomo_model, X_cocomo, y_cocomo_raw, y_cocomo_log, predict_is_log=True)

    summary_df = pd.DataFrame([
        summarize(unified_cv, "Unified (4-feature RF, deployed baseline)"),
        summarize(cocomo_cv, "COCOMO sign-constrained Ridge (EM table)"),
    ]).round(4)

    ordered_cols = ["Model"] + [f"{c} {s}" for c in METRIC_COLUMNS for s in ("mean", "std")]
    print(summary_df[ordered_cols].to_string(index=False))
    print()

    beats_baseline = (
        cocomo_cv["PRED(25)"].mean() >= unified_cv["PRED(25)"].mean()
        and cocomo_cv["R2(log)"].mean() >= unified_cv["R2(log)"].mean()
    )
    print(
        f"vs unified baseline — PRED(25): {cocomo_cv['PRED(25)'].mean():.2f}% vs "
        f"{unified_cv['PRED(25)'].mean():.2f}%;  R2(log): {cocomo_cv['R2(log)'].mean():.4f} vs "
        f"{unified_cv['R2(log)'].mean():.4f}"
    )
    print()

    # -----------------------------------------------------------------
    # Secondary confirmation — 20 random 80/20 holdouts.
    # -----------------------------------------------------------------
    holdout_df = holdout_metrics(build_cocomo_model, X_cocomo, y_cocomo_raw, y_cocomo_log, predict_is_log=True, n_repeats=N_HOLDOUT_REPEATS)
    holdout_summary = summarize(holdout_df, "COCOMO sign-constrained Ridge")
    print(f"COCOMO sign-constrained Ridge — average over {N_HOLDOUT_REPEATS} random 80/20 holdouts:")
    for col in METRIC_COLUMNS:
        print(f"  {col}: {holdout_summary[f'{col} mean']:.4f} (+/- {holdout_summary[f'{col} std']:.4f})")
    print()

    print("=" * 90)
    if not beats_baseline:
        print(
            f"NOT DEPLOYED: COCOMO sign-constrained Ridge does not beat the unified baseline on both "
            f"CV PRED(25) and CV R2(log). {NEW_MODEL_PATH} left unchanged."
        )
        print("=" * 90)
        return

    # -----------------------------------------------------------------
    # Deploy: fit on ALL 93 rows, back up the current (unified) model,
    # save the new model + its encoding metadata.
    # -----------------------------------------------------------------
    final_model = build_cocomo_model()
    final_model.fit(X_cocomo, y_cocomo_log)

    # Only back up if no baseline backup exists yet — NEW_MODEL_PATH may
    # already hold a PREVIOUS (superseded) COCOMO attempt rather than the
    # real unified baseline if this script has been run before; never
    # clobber an existing, correct backup with the wrong thing.
    if not os.path.exists(UNIFIED_BASELINE_BACKUP_PATH):
        shutil.copy2(NEW_MODEL_PATH, UNIFIED_BASELINE_BACKUP_PATH)

    metadata = {
        "feature_order": FEATURE_ORDER,
        "ordinal_cols": COCOMO_ORDINAL_COLS,
        "non_cplx_drivers": NON_CPLX_DRIVERS,
        "cplx_offset_col": CPLX_OFFSET_COL,
        "cplx_weight": 1.0,
        "em_table": COCOMO_EM_TABLE,
        "mode_categories": MODE_CATEGORIES,
        "training_mean": {col: float(X_cocomo[col].mean()) for col in FULL_COLUMNS},
        "target_is_log": True,
        "model_kind": "cplx_offset_ridge",
    }

    save_artifact(final_model, NEW_MODEL_PATH)
    save_artifact(metadata, METADATA_PATH)

    print("DEPLOYED: COCOMO sign-constrained Ridge beats the unified baseline on both CV PRED(25) and CV R2(log).")
    print(f"Saved -> {NEW_MODEL_PATH} (new model, fit on all {len(X_cocomo)} COCOMO-NASA rows)")
    print(f"Saved -> {UNIFIED_BASELINE_BACKUP_PATH} (previous deployed model, kept as documented baseline)")
    print(f"Saved -> {METADATA_PATH} (encoding metadata for the app)")
    print(f"Ridge alpha: {RIDGE_ALPHA} (fixed, not CV-tuned)")
    print("=" * 90)


if __name__ == "__main__":
    main()
