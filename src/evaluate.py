"""
EstimateX — Step 7: model evaluation.

Loads the 8 models saved by src/train_models.py and the test sets saved
by src/split_data.py, scores every model/dataset combination on the test
set (never touched during training or CV), and writes the comparison
table + MAE bar charts that feed directly into the paper's Results
section.

No feature importance here — that's Step 8.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from split_data import DATASET_CONFIGS, split_features_target
from train_models import DATASET_SLUGS, MODEL_NAMES
from preprocessing import load_artifact

MODELS_DIR = "models"
RESULTS_DIR = "results"

# Display-friendly model names for the printed table / chart axis labels.
MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svr": "SVR",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
# MAE and RMSE are built into scikit-learn, so they're just thin wrappers
# below. MMRE and PRED(25) are standard in the software effort-estimation
# literature (Conte et al., 1986) but are NOT in scikit-learn, so they're
# implemented from scratch here.

def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error = sqrt(mean((y_true - y_pred)^2)).
    A thin wrapper around sklearn's mean_squared_error (sqrt taken
    manually rather than relying on a `squared=` kwarg, since that
    parameter has been deprecated/removed across recent sklearn versions).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mmre(y_true, y_pred) -> float:
    """Mean Magnitude of Relative Error.

    For each prediction, the relative error is:
        MRE_i = |y_true_i - y_pred_i| / y_true_i
    MMRE is the mean of MRE_i across all n test samples:
        MMRE = (1/n) * sum(MRE_i)

    Lower is better. Assumes y_true has no zero values (true here — both
    datasets' effort/cost targets are always positive).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    relative_errors = np.abs(y_true - y_pred) / y_true
    return float(relative_errors.mean())


def pred25(y_true, y_pred) -> float:
    """PRED(25): the percentage of predictions whose relative error
    (as defined in mmre() above) is within 25% of the actual value:
        PRED(25) = 100 * (count(MRE_i <= 0.25) / n)

    Higher is better (100 = every prediction within 25% of actual).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    relative_errors = np.abs(y_true - y_pred) / y_true
    within_25_pct = (relative_errors <= 0.25).sum()
    return float(100.0 * within_25_pct / len(y_true))


def compute_all_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MMRE": mmre(y_true, y_pred),
        "PRED(25)": pred25(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_test_set(test_path: str, target_col: str):
    test_df = pd.read_csv(test_path)
    return split_features_target(test_df, target_col)


def evaluate_dataset(name: str, test_path: str, target_col: str) -> list[dict]:
    """Load each of the 4 saved models for this dataset, predict on its
    test set, and compute all 4 metrics. Returns one row per model.
    """
    slug = DATASET_SLUGS[name]
    X_test, y_test = load_test_set(test_path, target_col)

    rows = []
    for model_name in MODEL_NAMES:
        model_path = f"{MODELS_DIR}/{slug}_{model_name}.pkl"
        model = load_artifact(model_path)
        y_pred = model.predict(X_test)
        metrics = compute_all_metrics(y_test, y_pred)
        rows.append(
            {
                "Dataset": name,
                "Model": MODEL_LABELS[model_name],
                **metrics,
            }
        )
    return rows


def summarize_best_models(results_df: pd.DataFrame) -> None:
    """Print, per dataset, which model wins on each metric individually,
    and call out the overall best model (primary criterion: lowest MAE,
    since it's in the target's real units and the most directly
    interpretable single number for the paper).
    """
    overall_best = {}

    for dataset in results_df["Dataset"].unique():
        subset = results_df[results_df["Dataset"] == dataset]

        best_mae = subset.loc[subset["MAE"].idxmin()]
        best_rmse = subset.loc[subset["RMSE"].idxmin()]
        best_mmre = subset.loc[subset["MMRE"].idxmin()]
        best_pred25 = subset.loc[subset["PRED(25)"].idxmax()]

        winners = {best_mae["Model"], best_rmse["Model"], best_mmre["Model"], best_pred25["Model"]}
        overall_best[dataset] = best_mae["Model"]

        print(f"--- {dataset} ---")
        print(f"  Lowest MAE:      {best_mae['Model']} (MAE={best_mae['MAE']:.2f})")
        print(f"  Lowest RMSE:     {best_rmse['Model']} (RMSE={best_rmse['RMSE']:.2f})")
        print(f"  Lowest MMRE:     {best_mmre['Model']} (MMRE={best_mmre['MMRE']:.3f})")
        print(f"  Highest PRED(25): {best_pred25['Model']} (PRED(25)={best_pred25['PRED(25)']:.1f}%)")

        if len(winners) == 1:
            print(f"  => {best_mae['Model']} wins on ALL four metrics — the clear best model for {dataset}.")
        else:
            print(
                f"  => No single model wins every metric; by primary metric (lowest MAE), "
                f"{best_mae['Model']} is the best-performing model on {dataset}."
            )
        print()

    datasets = list(overall_best.keys())
    if len(datasets) == 2:
        model_a, model_b = overall_best[datasets[0]], overall_best[datasets[1]]
        if model_a == model_b:
            print(
                f"Same algorithm wins on both datasets: {model_a} is best on both "
                f"{datasets[0]} and {datasets[1]} (by MAE)."
            )
        else:
            print(
                f"Different algorithms win on each dataset: {model_a} is best on "
                f"{datasets[0]}, while {model_b} is best on {datasets[1]} (by MAE) — "
                "worth discussing in the paper as evidence that no single model "
                "generalizes best across both effort-estimation domains."
            )


def plot_mae_comparison(results_df: pd.DataFrame, dataset: str, output_path: str) -> None:
    subset = results_df[results_df["Dataset"] == dataset].sort_values("MAE")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(subset["Model"], subset["MAE"], color=colors[: len(subset)])

    for bar, value in zip(bars, subset["MAE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(f"{dataset}: Test-set MAE by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("EstimateX — Step 7 model evaluation (test set)")
    print()

    all_rows = []
    for cfg in DATASET_CONFIGS:
        all_rows.extend(
            evaluate_dataset(name=cfg["name"], test_path=cfg["test_path"], target_col=cfg["target_col"])
        )

    results_df = pd.DataFrame(all_rows)
    # Sort by dataset, then within each dataset from best to worst by MAE
    # (ascending — lower MAE is better). MAE is chosen as the primary sort
    # key since it's in the target's real units and the single most
    # commonly cited headline metric; PRED(25) winners are still called
    # out explicitly per-metric in the summary below.
    results_df = results_df.sort_values(["Dataset", "MAE"], ascending=[True, True]).reset_index(drop=True)
    results_df_rounded = results_df.round(3)

    print("=" * 78)
    print("Combined model comparison — test set")
    print("=" * 78)
    print(results_df_rounded.to_string(index=False))
    print()

    comparison_path = f"{RESULTS_DIR}/model_comparison.csv"
    results_df_rounded.to_csv(comparison_path, index=False)
    print(f"Saved -> {comparison_path}")
    print()

    print("=" * 78)
    print("Best-performing model per dataset")
    print("=" * 78)
    summarize_best_models(results_df)
    print()

    cocomo_chart_path = f"{RESULTS_DIR}/mae_comparison_cocomo.png"
    desharnais_chart_path = f"{RESULTS_DIR}/mae_comparison_desharnais.png"
    plot_mae_comparison(results_df, "COCOMO-NASA", cocomo_chart_path)
    plot_mae_comparison(results_df, "Desharnais", desharnais_chart_path)
    print(f"Saved -> {cocomo_chart_path}")
    print(f"Saved -> {desharnais_chart_path}")
