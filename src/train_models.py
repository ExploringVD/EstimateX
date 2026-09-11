"""
EstimateX — Step 6: model training.

Trains four regressors (Linear Regression, Random Forest, XGBoost, SVR) on
each of the two datasets (COCOMO-NASA, Desharnais), using the train/test
splits already produced by src/split_data.py — this module never re-splits
or re-preprocesses, it only loads what Step 5 already saved.

Because both datasets are small (under 200 records combined), 5-fold
cross-validation on the training set is used to sanity-check how stable
each model is before Step 7's full test-set evaluation (MAE/RMSE/MMRE/
PRED(25)), rather than trusting a single train/test score. No
hyperparameter tuning happens here — defaults only, with RANDOM_STATE
(imported from split_data, the single source of truth) applied everywhere
a model accepts one, for reproducibility.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from xgboost import XGBRegressor

from split_data import DATASET_CONFIGS, RANDOM_STATE, split_features_target
from preprocessing import save_artifact

MODELS_DIR = "models"
CV_FOLDS = 5

# Display name -> filename slug used for saved models, e.g.
# models/cocomo_nasa_random_forest.pkl
DATASET_SLUGS = {
    "COCOMO-NASA": "cocomo_nasa",
    "Desharnais": "desharnais",
}

# Single source of truth for which model slugs exist and in what order —
# src/evaluate.py reuses this to know which 4 saved .pkl files to load per
# dataset, instead of hardcoding the list a second time.
MODEL_NAMES = ["linear_regression", "random_forest", "xgboost", "svr"]


def build_model_factories(random_state: int = RANDOM_STATE) -> dict:
    """Fresh, untrained model instances. A factory (not a shared dict of
    already-constructed models) because cross_val_score/refit need a new,
    unfitted estimator each time — reusing one fitted instance across
    folds would leak state between them.

    Reasonable defaults only, no tuning:
    - LinearRegression: plain baseline, no hyperparameters to set.
    - RandomForestRegressor: default n_estimators=100, random_state fixed.
    - XGBRegressor: default n_estimators=100, random_state fixed.
    - SVR: sklearn's SVR has no random_state parameter (its solution is a
      deterministic convex optimization, no internal randomness), so
      random_state is intentionally not passed here.
    """
    return {
        "linear_regression": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(random_state=random_state),
        "xgboost": lambda: XGBRegressor(random_state=random_state, objective="reg:squarederror"),
        "svr": lambda: SVR(),
    }


def load_train_test(train_path: str, test_path: str, target_col: str):
    """Load the already-split train/test CSVs from Step 5 and separate
    features from the target. No splitting or preprocessing happens here.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

    return X_train, X_test, y_train, y_test


def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """5-fold CV on the training set only. Shuffled with a fixed
    random_state so folds are reproducible across reruns. Reports both R²
    (how well the model explains variance — the headline comparison
    metric) and RMSE (in the target's real units) so instability shows up
    either way a fold happens to be easy or hard.
    """
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")
    neg_rmse_scores = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -neg_rmse_scores

    return {
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std(),
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
    }


def train_and_save_dataset(name: str, train_path: str, test_path: str, target_col: str) -> list[dict]:
    """Cross-validate, fit, and save all four models for one dataset.
    Returns one summary row per model for the final comparison table.
    """
    slug = DATASET_SLUGS[name]
    X_train, X_test, y_train, y_test = load_train_test(train_path, test_path, target_col)

    print("=" * 70)
    print(f"{name}  (X_train={X_train.shape}, X_test={X_test.shape})")
    print("-" * 70)

    factories = build_model_factories()
    rows = []

    for model_name, factory in factories.items():
        cv_result = cross_validate_model(factory(), X_train, y_train)

        print(f"[{model_name}]")
        print(
            f"  CV R^2:   mean={cv_result['cv_r2_mean']:.4f}  std={cv_result['cv_r2_std']:.4f}"
        )
        print(
            f"  CV RMSE:  mean={cv_result['cv_rmse_mean']:.4f}  std={cv_result['cv_rmse_std']:.4f}"
        )

        # Fit a fresh instance on the FULL training set (not any single CV
        # fold) — this is the model that actually gets saved and used in
        # Step 7's test-set evaluation.
        final_model = factory()
        final_model.fit(X_train, y_train)

        model_path = f"{MODELS_DIR}/{slug}_{model_name}.pkl"
        save_artifact(final_model, model_path)
        print(f"  Saved -> {model_path}")
        print()

        rows.append(
            {
                "dataset": name,
                "model": model_name,
                "cv_r2_mean": cv_result["cv_r2_mean"],
                "cv_r2_std": cv_result["cv_r2_std"],
                "cv_rmse_mean": cv_result["cv_rmse_mean"],
                "cv_rmse_std": cv_result["cv_rmse_std"],
            }
        )

    return rows


if __name__ == "__main__":
    print("EstimateX — Step 6 model training")
    print(f"(cv_folds={CV_FOLDS}, random_state={RANDOM_STATE}, no hyperparameter tuning)")
    print()

    all_rows = []
    for cfg in DATASET_CONFIGS:
        all_rows.extend(
            train_and_save_dataset(
                name=cfg["name"],
                train_path=cfg["train_path"],
                test_path=cfg["test_path"],
                target_col=cfg["target_col"],
            )
        )

    summary_df = pd.DataFrame(all_rows)
    summary_df = summary_df.round(4)

    print("=" * 70)
    print("Summary — mean CV score per model (sanity check before Step 7)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
