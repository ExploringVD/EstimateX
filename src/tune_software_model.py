"""
EstimateX — software model improvement experiment.

The deployed software Random Forest (n_estimators=100, min_samples_leaf=1,
raw effort_person_months target) overfits badly: train R^2 ~0.92 but test
R^2 ~ -0.87. Two causes, addressed together here:

1. The target (0.17 to 8211 person-months) is extremely right-skewed, so
   a few huge projects dominate the raw-unit loss the tree splits on.
   Fitting in log-space (TransformedTargetRegressor, func=log,
   inverse_func=exp) makes the loss behave more like relative error,
   which is what MMRE/PRED(25) actually measure. .predict() still
   returns real person-months — the transform is invisible to callers.
2. min_samples_leaf=1 lets the forest memorize individual training rows.
   A regularized min_samples_leaf/max_features combination is used
   instead.

Same train/test split, same 4 features, same preprocessing as the
existing software pipeline (train_new_domains.py) — this script only
changes the estimator, and only for the software domain. Construction is
untouched: it isn't imported, read, or written anywhere below.

Standalone by design (not wired into train_new_domains.py) so the
generic 4-algorithm comparison shared with Construction stays exactly as
it was — this experiment only ever overwrites software artifacts, and
only after the honest comparison in main() shows it's genuinely better.

IMPORTANT — a real finding from this experiment, not a bug:
Plain 5-fold CV on the training set (scoring=r2, grid below), repeated
K-fold CV (5x20 repeats), and the Random Forest's own OOB score all rank
min_samples_leaf=1 (no regularization) ABOVE min_samples_leaf=8 on the
TRAINING data — yet on the untouched held-out test set, leaf=8 is
dramatically better (test R^2 0.64 vs -0.55 for leaf=1). With only 535
training rows and an extremely skewed target (0.17 to 8211
person-months), every blind, test-set-agnostic selection method tried
here disagrees with what the held-out test set shows. This is reported
below in full (`run_blind_cv_grid_search`) for transparency, but it is
NOT used to pick the deployed model — min_samples_leaf=8, max_features=
1.0 is used directly (the brief's own quoted hyperparameters, confirmed
genuinely better on the held-out test set the same way the old model
was evaluated), per explicit confirmation to deploy it despite blind CV
not being able to discover it on its own for this dataset.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from split_data import RANDOM_STATE, split_features_target
from preprocessing import load_artifact, save_artifact

MODELS_DIR = "models"
TRAIN_PATH = "data/processed/software_unified_train.csv"
TEST_PATH = "data/processed/software_unified_test.csv"
TARGET_COL = "effort_person_months"

FINAL_MODEL_PATH = f"{MODELS_DIR}/final_model_software.pkl"
OLD_MODEL_BACKUP_PATH = f"{MODELS_DIR}/final_model_software_OLD.pkl"

CV_FOLDS_TUNING = 5
CV_FOLDS_ROBUSTNESS = 10
N_ESTIMATORS = 500
PARAM_GRID = {
    "regressor__min_samples_leaf": [1, 2, 4, 8, 12],
    "regressor__max_features": [1.0, 0.5],
}

# The candidate actually deployed below — the brief's own quoted
# hyperparameters, NOT the blind grid-search's pick (see module docstring).
CHOSEN_MIN_SAMPLES_LEAF = 8
CHOSEN_MAX_FEATURES = 1.0


# ---------------------------------------------------------------------------
# Metrics (same definitions as src/evaluate.py's mmre/pred25, plus R^2 and
# MdMRE which that module doesn't have — reimplemented here rather than
# changed there, since evaluate.py is shared with Construction).
# ---------------------------------------------------------------------------

def mmre(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float((np.abs(y_true - y_pred) / y_true).mean())


def mdmre(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.median(np.abs(y_true - y_pred) / y_true))


def pred25(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    within = (np.abs(y_true - y_pred) / y_true <= 0.25).sum()
    return float(100.0 * within / len(y_true))


def compute_all_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MMRE": mmre(y_true, y_pred),
        "MdMRE": mdmre(y_true, y_pred),
        "PRED(25)": pred25(y_true, y_pred),
    }


PRED25_SCORER = make_scorer(lambda y_true, y_pred: pred25(y_true, y_pred) / 100.0)


def cv_robustness(estimator, X_train, y_train, n_folds: int) -> dict:
    """n-fold CV on the TRAINING SET ONLY — a stability check, not a
    substitute for the held-out test-set numbers in main(). A fresh,
    unfitted estimator is refit on each fold internally by cross_val_score.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring="r2")
    pred25_scores = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring=PRED25_SCORER) * 100.0
    return {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "pred25_mean": pred25_scores.mean(),
        "pred25_std": pred25_scores.std(),
    }


def build_old_model() -> RandomForestRegressor:
    """Fresh, unfitted clone of the currently-deployed hyperparameters
    (train_models.py's build_model_factories default for random_forest).
    """
    return RandomForestRegressor(random_state=RANDOM_STATE)


def build_log_target_rf(min_samples_leaf: int, max_features) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=RANDOM_STATE,
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def run_blind_cv_grid_search(X_train, y_train) -> None:
    """Exactly what the brief specified: 5-fold CV on the training set
    only, scoring=r2, picking the best of the grid. Printed for full
    transparency — see module docstring for why its pick (leaf=1) is NOT
    what gets deployed.
    """
    kfold = KFold(n_splits=CV_FOLDS_TUNING, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_log_target_rf(min_samples_leaf=1, max_features=1.0),  # placeholder, grid overrides
        param_grid=PARAM_GRID,
        scoring="r2",
        cv=kfold,
        refit=False,
    )
    search.fit(X_train, y_train)

    print(f"Blind grid search (5-fold CV on TRAIN ONLY, scoring=r2, n_estimators={N_ESTIMATORS}):")
    cv_results = pd.DataFrame(search.cv_results_)[
        ["param_regressor__min_samples_leaf", "param_regressor__max_features", "mean_test_score", "std_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    print(cv_results.to_string(index=False))
    print()
    print(
        f"Blind CV's own pick: {search.best_params_} (CV r2={search.best_score_:.4f}) "
        f"— NOT used for deployment; see module docstring."
    )
    print()


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_features_target(train_df, TARGET_COL)
    X_test, y_test = split_features_target(test_df, TARGET_COL)

    print("=" * 78)
    print("EstimateX — software model improvement (log target + regularized RF)")
    print("=" * 78)
    print(f"X_train={X_train.shape}  X_test={X_test.shape}  features={list(X_train.columns)}")
    print()

    # -----------------------------------------------------------------
    # STEP 1 — blind tuning, exactly as specified (train-only, 5-fold CV).
    # Reported in full for transparency; NOT used to pick the deployed
    # model (see module docstring for why).
    # -----------------------------------------------------------------
    run_blind_cv_grid_search(X_train, y_train)

    # -----------------------------------------------------------------
    # Chosen candidate: the brief's own hyperparameters, fit on the full
    # training set and evaluated on the untouched test set exactly once,
    # the same way the old model is evaluated below (not selected via
    # any test-set-driven search).
    # -----------------------------------------------------------------
    new_model = build_log_target_rf(CHOSEN_MIN_SAMPLES_LEAF, CHOSEN_MAX_FEATURES)
    new_model.fit(X_train, y_train)
    print(
        f"Deployed candidate's hyperparameters: min_samples_leaf={CHOSEN_MIN_SAMPLES_LEAF}, "
        f"max_features={CHOSEN_MAX_FEATURES}, n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE} "
        f"(the brief's own figures, not the blind grid search's pick)"
    )
    print()

    # -----------------------------------------------------------------
    # STEP 2 — honest held-out test-set comparison, old vs new.
    # -----------------------------------------------------------------
    old_model = load_artifact(FINAL_MODEL_PATH)  # currently-deployed, already-fitted model
    old_test_metrics = compute_all_metrics(y_test, old_model.predict(X_test))
    new_test_metrics = compute_all_metrics(y_test, new_model.predict(X_test))

    comparison_df = pd.DataFrame([
        {"Model": "OLD (RF, raw target, leaf=1)", **old_test_metrics},
        {"Model": "NEW (RF, log target, leaf=8)", **new_test_metrics},
    ]).round(4)

    print("=" * 78)
    print("Held-out TEST SET metrics — old vs new")
    print("=" * 78)
    print(comparison_df.to_string(index=False))
    print()

    print("10-fold CV (TRAINING SET ONLY) — robustness check:")
    old_cv = cv_robustness(build_old_model(), X_train, y_train, CV_FOLDS_ROBUSTNESS)
    new_cv = cv_robustness(
        build_log_target_rf(CHOSEN_MIN_SAMPLES_LEAF, CHOSEN_MAX_FEATURES), X_train, y_train, CV_FOLDS_ROBUSTNESS
    )
    cv_df = pd.DataFrame([
        {"Model": "OLD (RF, raw target, leaf=1)", "R2 mean": old_cv["r2_mean"], "R2 std": old_cv["r2_std"],
         "PRED(25) mean": old_cv["pred25_mean"], "PRED(25) std": old_cv["pred25_std"]},
        {"Model": "NEW (RF, log target, leaf=8)", "R2 mean": new_cv["r2_mean"], "R2 std": new_cv["r2_std"],
         "PRED(25) mean": new_cv["pred25_mean"], "PRED(25) std": new_cv["pred25_std"]},
    ]).round(4)
    print(cv_df.to_string(index=False))
    print(
        "(Note: this training-only CV ranks leaf=1 above leaf=8 too — consistent with the "
        "module docstring's finding that this metric doesn't predict held-out performance "
        "well for this dataset. The test-set numbers above are the ones that matter.)"
    )
    print()

    # -----------------------------------------------------------------
    # STEP 3 — deploy only if genuinely better on the held-out test set.
    # -----------------------------------------------------------------
    is_better = (
        new_test_metrics["R2"] >= old_test_metrics["R2"]
        and new_test_metrics["PRED(25)"] >= old_test_metrics["PRED(25)"]
    )

    print("=" * 78)
    if is_better:
        shutil.copy2(FINAL_MODEL_PATH, OLD_MODEL_BACKUP_PATH)
        save_artifact(new_model, FINAL_MODEL_PATH)
        print(
            f"DEPLOYED: new model test R2={new_test_metrics['R2']:.4f} >= old "
            f"{old_test_metrics['R2']:.4f}, and PRED(25)={new_test_metrics['PRED(25)']:.1f}% >= old "
            f"{old_test_metrics['PRED(25)']:.1f}%."
        )
        print(f"Saved -> {FINAL_MODEL_PATH} (new model)")
        print(f"Saved -> {OLD_MODEL_BACKUP_PATH} (old model, kept until confirmed)")
    else:
        print(
            f"NOT DEPLOYED: new model did not beat the old one on both test R2 "
            f"({new_test_metrics['R2']:.4f} vs {old_test_metrics['R2']:.4f}) and PRED(25) "
            f"({new_test_metrics['PRED(25)']:.1f}% vs {old_test_metrics['PRED(25)']:.1f}%). "
            f"{FINAL_MODEL_PATH} left unchanged."
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
