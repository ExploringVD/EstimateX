"""Trains and deploys the software effort model on COCOMO-NASA (93 projects); run from the project root.

log(effort) = log(CPLX multiplier) + Ridge over the other 14 drivers' log(Boehm-1981 multiplier),
log(KLOC) and development mode. Also stores the CV-based effort likely range in the metadata.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from preprocessing import save_artifact

RAW_COCOMO_PATH = "data/raw/cocomo_nasa.csv"
COCOMO_TARGET_COL = "act_effort"
KLOC_COL = "equivphyskloc"
MODE_COL = "mode"
MODE_CATEGORIES = ["embedded", "organic", "semidetached"]

MODEL_PATH = "models/final_model_software.pkl"
METADATA_PATH = "models/final_model_software_metadata.pkl"

CV_FOLDS = 10
CV_REPEATS = list(range(10))  # KFold seeds 0-9
RIDGE_ALPHA = 0.1

COCOMO_ORDINAL_SCALE = ["vl", "l", "n", "h", "vh", "xh"]
COCOMO_ORDINAL_COLS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]

# Boehm 1981 Intermediate COCOMO effort multipliers; only officially defined levels are listed.
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

CPLX_OFFSET_COL = "cplx_log_em"
NON_CPLX_DRIVERS = [c for c in COCOMO_ORDINAL_COLS if c != "cplx"]
FEATURE_ORDER = NON_CPLX_DRIVERS + ["log_kloc"] + [f"mode_{c}" for c in MODE_CATEGORIES]
FULL_COLUMNS = FEATURE_ORDER + [CPLX_OFFSET_COL]

EFFORT_RANGE_PERCENTILES = (25, 75)
EFFORT_RANGE_Q25_KEY = "cv_log_residual_q25"
EFFORT_RANGE_Q75_KEY = "cv_log_residual_q75"


def get_em(driver: str, level: str) -> float:
    """Published multiplier for `level`, or for the nearest level defined for this driver."""
    defined = COCOMO_EM_TABLE[driver]
    if level in defined:
        return defined[level]
    idx = COCOMO_ORDINAL_SCALE.index(level)
    return defined[min(defined, key=lambda lv: abs(COCOMO_ORDINAL_SCALE.index(lv) - idx))]


def build_cocomo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in NON_CPLX_DRIVERS:
        out[col] = df[col].apply(lambda level, driver=col: np.log(get_em(driver, level)))
    out["log_kloc"] = np.log(df[KLOC_COL])
    mode_dummies = pd.get_dummies(df[MODE_COL], prefix="mode")
    for cat in MODE_CATEGORIES:
        out[f"mode_{cat}"] = mode_dummies.get(f"mode_{cat}", 0)
    out[CPLX_OFFSET_COL] = df["cplx"].apply(lambda level: np.log(get_em("cplx", level)))
    return out


def pred_at(y_true, y_pred, threshold_pct: float) -> float:
    mre = np.abs(y_true - y_pred) / y_true
    return float(100.0 * (mre <= threshold_pct / 100.0).sum() / len(y_true))


def compute_report_metrics(y_true_raw, y_pred_raw) -> dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    mre = np.abs(y_true_raw - y_pred_raw) / y_true_raw
    return {
        "PRED(25)": pred_at(y_true_raw, y_pred_raw, 25),
        "PRED(50)": pred_at(y_true_raw, y_pred_raw, 50),
        "R2(log)": r2_score(np.log(y_true_raw), np.log(np.clip(y_pred_raw, 1e-6, None))),
        "R2(raw)": r2_score(y_true_raw, y_pred_raw),
        "MMRE": float(mre.mean()),
        "MdMRE": float(np.median(mre)),
    }


METRIC_COLUMNS = ["PRED(25)", "PRED(50)", "R2(log)", "R2(raw)", "MMRE", "MdMRE"]


class CplxOffsetRidge(BaseEstimator, RegressorMixin):
    """Ridge(positive=True) on log(effort) with cplx as a fixed weight-1 offset."""

    # log(EM) already encodes each driver's direction, so non-negative weights keep it correct;
    # a fixed cplx offset keeps complexity monotonic, which a free weight was not under collinearity.
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
        return self.ridge_.predict(X[self.feature_names_]) + X[CPLX_OFFSET_COL].to_numpy()


# Pickle the class under its importable module name even when this file runs as __main__,
# so the app can unpickle the saved model.
sys.modules.setdefault("train_software_cocomo", sys.modules[__name__])
CplxOffsetRidge.__module__ = "train_software_cocomo"


def build_cocomo_model() -> CplxOffsetRidge:
    return CplxOffsetRidge(alpha=RIDGE_ALPHA)


def oof_predictions(X, y_log, seed: int) -> np.ndarray:
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(build_cocomo_model(), X, y_log, cv=kfold)


def repeated_cv_metrics(X, y_raw) -> pd.DataFrame:
    """One row of metrics per CV repeat, from that repeat's pooled out-of-fold predictions."""
    y_log = np.log(y_raw)
    return pd.DataFrame([compute_report_metrics(y_raw, np.exp(oof_predictions(X, y_log, s))) for s in CV_REPEATS])


def effort_range_quantiles(X, y_log) -> dict:
    """25th/75th percentiles of out-of-fold log(actual) - log(predicted), pooled over all repeats."""
    residuals = np.concatenate([np.asarray(y_log) - oof_predictions(X, y_log, s) for s in CV_REPEATS])
    q25, q75 = np.percentile(residuals, EFFORT_RANGE_PERCENTILES)
    return {EFFORT_RANGE_Q25_KEY: float(q25), EFFORT_RANGE_Q75_KEY: float(q75)}


def nested_range_coverage(X: pd.DataFrame, y_log: pd.Series) -> float:
    """% of held-out projects inside a range computed only from their training folds."""
    inside = []
    for seed in CV_REPEATS:
        outer = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in outer.split(X):
            X_train, y_train = X.iloc[train_idx], y_log.iloc[train_idx]
            inner = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
            inner_resid = y_train.to_numpy() - cross_val_predict(build_cocomo_model(), X_train, y_train, cv=inner)
            q25, q75 = np.percentile(inner_resid, EFFORT_RANGE_PERCENTILES)
            model = build_cocomo_model().fit(X_train, y_train)
            test_resid = y_log.iloc[test_idx].to_numpy() - model.predict(X.iloc[test_idx])
            inside.extend((test_resid >= q25) & (test_resid <= q75))
    return float(np.mean(inside) * 100)


def main() -> None:
    df = pd.read_csv(RAW_COCOMO_PATH)
    X = build_cocomo_features(df)[FULL_COLUMNS]
    y_raw = df[COCOMO_TARGET_COL]
    y_log = np.log(y_raw)

    cv = repeated_cv_metrics(X, y_raw)
    print(f"Software (COCOMO-NASA, {len(X)} rows) — {CV_FOLDS}-fold CV x {len(CV_REPEATS)} repeats:")
    for col in METRIC_COLUMNS:
        print(f"  {col:9s}: {cv[col].mean():8.3f}  (+/- {cv[col].std():.3f})")

    model = build_cocomo_model().fit(X, y_log)
    effort_range = effort_range_quantiles(X, y_log)
    metadata = {
        "feature_order": FEATURE_ORDER,
        "ordinal_cols": COCOMO_ORDINAL_COLS,
        "non_cplx_drivers": NON_CPLX_DRIVERS,
        "cplx_offset_col": CPLX_OFFSET_COL,
        "cplx_weight": 1.0,
        "em_table": COCOMO_EM_TABLE,
        "mode_categories": MODE_CATEGORIES,
        "training_mean": {col: float(X[col].mean()) for col in FULL_COLUMNS},
        "target_is_log": True,
        "model_kind": "cplx_offset_ridge",
        **effort_range,
    }
    save_artifact(model, MODEL_PATH)
    save_artifact(metadata, METADATA_PATH)

    q25, q75 = effort_range[EFFORT_RANGE_Q25_KEY], effort_range[EFFORT_RANGE_Q75_KEY]
    print(f"Likely range: x{np.exp(q25):.3f} .. x{np.exp(q75):.3f} of predicted effort "
          f"(q25={q25:+.4f}, q75={q75:+.4f}); nested-CV coverage {nested_range_coverage(X, y_log):.1f}%")
    print(f"Saved -> {MODEL_PATH}, {METADATA_PATH}")


if __name__ == "__main__":
    main()
