"""Trains and deploys the construction cost model.

Compares Linear Regression, Random Forest, XGBoost and SVR on a fixed 80/20 split and deploys
the best tree-based model with its scaler and encoder. Run from the project root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor

from preprocess_construction import TARGET_COL, load_construction_raw, preprocess_construction
from preprocessing import save_artifact

PROCESSED_DIR = "data/processed"
CLEAN_PATH = f"{PROCESSED_DIR}/construction_clean.csv"
TRAIN_PATH = f"{PROCESSED_DIR}/construction_train.csv"
TEST_PATH = f"{PROCESSED_DIR}/construction_test.csv"
MODELS_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAE_TOLERANCE_PCT = 5.0  # a PRED(25) leader may win if its MAE is within this % of the best MAE

MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE),
    "XGBoost": lambda: XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
    "SVR": lambda: SVR(),
}
# The app explains construction predictions with feature_importances_, so only these can deploy.
TREE_MODELS = {"Random Forest", "XGBoost"}


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def save_split(clean_path: str) -> None:
    X, y = split_xy(pd.read_csv(clean_path))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for path, X_part, y_part in ((TRAIN_PATH, X_train, y_train), (TEST_PATH, X_test, y_test)):
        part = X_part.copy()
        part[TARGET_COL] = y_part
        part.to_csv(path, index=False)


def test_metrics(y_true, y_pred) -> dict:
    mre = np.abs(np.asarray(y_true, dtype=float) - y_pred) / np.asarray(y_true, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MMRE": float(mre.mean()),
        "PRED(25)": float(100.0 * (mre <= 0.25).sum() / len(mre)),
    }


def pick_combined_winner(results: pd.DataFrame) -> str:
    """Lowest MAE wins, unless the PRED(25) leader's MAE is within MAE_TOLERANCE_PCT of it."""
    best_mae = results.loc[results["MAE"].idxmin()]
    best_pred = results.loc[results["PRED(25)"].idxmax()]
    if best_mae["Model"] == best_pred["Model"]:
        return best_mae["Model"]
    mae_gap_pct = (best_pred["MAE"] - best_mae["MAE"]) / best_mae["MAE"] * 100
    return best_pred["Model"] if mae_gap_pct <= MAE_TOLERANCE_PCT else best_mae["Model"]


def main() -> None:
    clean_df, encoder, scaler = preprocess_construction(load_construction_raw())
    clean_df.to_csv(CLEAN_PATH, index=False)
    save_split(CLEAN_PATH)

    X_train, y_train = split_xy(pd.read_csv(TRAIN_PATH))
    X_test, y_test = split_xy(pd.read_csv(TEST_PATH))
    print(f"Construction: X_train={X_train.shape}  X_test={X_test.shape}")

    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows = {}, []
    for name, factory in MODEL_FACTORIES.items():
        cv_r2 = cross_val_score(factory(), X_train, y_train, cv=kfold, scoring="r2")
        model = factory().fit(X_train, y_train)
        fitted[name] = model
        rows.append({"Model": name, **test_metrics(y_test, model.predict(X_test)), "CV R2": cv_r2.mean()})

    results = pd.DataFrame(rows)
    print(results.round(3).to_string(index=False))

    winner = pick_combined_winner(results)
    if winner not in TREE_MODELS:
        winner = pick_combined_winner(results[results["Model"].isin(TREE_MODELS)])
    print(f"Deploying: {winner}")

    save_artifact(fitted[winner], f"{MODELS_DIR}/final_model_construction.pkl")
    save_artifact(scaler, f"{MODELS_DIR}/final_scaler_construction.pkl")
    save_artifact(encoder, f"{MODELS_DIR}/final_encoder_construction.pkl")
    print(f"Saved -> {MODELS_DIR}/final_model_construction.pkl, final_scaler_construction.pkl, final_encoder_construction.pkl")


if __name__ == "__main__":
    main()
