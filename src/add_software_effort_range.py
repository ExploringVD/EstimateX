"""
EstimateX — adds the software effort "likely range" percentiles to the
EXISTING models/final_model_software_metadata.pkl.

Does NOT refit, re-save, or touch models/final_model_software.pkl, and
leaves every existing metadata key exactly as it was — it only adds (or
refreshes) the two keys from train_software_cocomo.effort_range_quantiles:
the 25th/75th percentiles of the model's out-of-fold log residuals,
log(actual) - log(predicted), from the same 10x repeated 10-fold CV used
everywhere else for this model.

Also prints the measured coverage (share of actual efforts inside
predicted x exp(q25) .. predicted x exp(q75)) two ways:
  - pooled: on the same OOF residuals the percentiles came from — ~50%
    by construction, a sanity check of the bookkeeping only.
  - nested: for each outer test fold, the band is built ONLY from inner-CV
    residuals on that fold's training rows — an honest out-of-sample
    check of whether the band really holds about half of new projects.

Run from the project root:   python src/add_software_effort_range.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

import train_software_cocomo as sw
from preprocessing import load_artifact, save_artifact


def nested_coverage(X: pd.DataFrame, y_log: pd.Series) -> float:
    inside = []
    for seed in sw.CV_REPEATS:
        outer = KFold(n_splits=sw.CV_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in outer.split(X):
            X_train, y_train = X.iloc[train_idx], y_log.iloc[train_idx]
            inner = KFold(n_splits=sw.CV_FOLDS, shuffle=True, random_state=seed)
            inner_resid = y_train.to_numpy() - cross_val_predict(sw.build_cocomo_model(), X_train, y_train, cv=inner)
            q25, q75 = np.percentile(inner_resid, sw.EFFORT_RANGE_PERCENTILES)
            model = sw.build_cocomo_model().fit(X_train, y_train)
            test_resid = y_log.iloc[test_idx].to_numpy() - model.predict(X.iloc[test_idx])
            inside.extend((test_resid >= q25) & (test_resid <= q75))
    return float(np.mean(inside) * 100)


def main() -> None:
    df = pd.read_csv(sw.RAW_COCOMO_PATH)
    X = sw.build_cocomo_features(df)[sw.FULL_COLUMNS]
    y_log = np.log(df[sw.COCOMO_TARGET_COL])

    quantiles = sw.effort_range_quantiles(X, y_log)
    q25, q75 = quantiles[sw.EFFORT_RANGE_Q25_KEY], quantiles[sw.EFFORT_RANGE_Q75_KEY]

    residuals = sw.oof_log_residuals(X, y_log)
    pooled = float(((residuals >= q25) & (residuals <= q75)).mean() * 100)
    nested = nested_coverage(X, y_log)

    metadata = load_artifact(sw.METADATA_PATH)
    before = {k: v for k, v in metadata.items() if k not in quantiles}
    metadata.update(quantiles)
    save_artifact(metadata, sw.METADATA_PATH)

    reloaded = load_artifact(sw.METADATA_PATH)
    unchanged = all(
        k in reloaded and repr(reloaded[k]) == repr(v) for k, v in before.items()
    )

    print("=" * 70)
    print("Software effort likely range — OOF log-residual percentiles")
    print("=" * 70)
    print(f"  residuals pooled: {len(residuals)} ({len(sw.CV_REPEATS)} repeats x {len(X)} rows)")
    print(f"  q25 = {q25:+.4f}  -> x{np.exp(q25):.3f} of predicted effort")
    print(f"  q75 = {q75:+.4f}  -> x{np.exp(q75):.3f} of predicted effort")
    print(f"  coverage, pooled OOF (by construction): {pooled:.1f}%")
    print(f"  coverage, nested CV (out-of-sample):    {nested:.1f}%")
    print(f"  existing metadata keys unchanged: {unchanged} ({len(before)} keys)")
    print(f"Saved -> {sw.METADATA_PATH} (added {', '.join(quantiles)}; model file not touched)")


if __name__ == "__main__":
    main()
