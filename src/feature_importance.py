"""
EstimateX — Step 8: feature importance & explainability.

Loads the 4 tree-based models (Random Forest and XGBoost, for both
datasets) saved by src/train_models.py, extracts their built-in
.feature_importances_, prints ranked importance tables, and saves one
horizontal bar chart per model.

Linear Regression and SVR are intentionally excluded — see the "Scoping
note" printed at the end and in results/feature_importance_summary.md:
neither exposes a comparable feature-importance API (Linear Regression's
coefficients aren't directly comparable across differently-scaled/encoded
features without extra work, and sklearn's SVR with the default RBF
kernel has no coefficient/importance concept at all), so explainability
in this project is currently tied to whichever tree-based model Step 9
picks as best.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from split_data import DATASET_CONFIGS
from train_models import DATASET_SLUGS
from preprocessing import load_artifact

MODELS_DIR = "models"
RESULTS_DIR = "results"

# The 2 tree-based model slugs this step covers, per dataset.
TREE_MODEL_NAMES = ["random_forest", "xgboost"]
MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}

# Chart readability: COCOMO-NASA expands to 48 one-hot features, which
# would be an unreadable bar chart, so charts show the top N only. The
# printed table (and the numbers behind the plain-language summary) still
# covers every feature, full ranking, no truncation.
CHART_TOP_N = 15

# Filename tags matching the exact paths requested in the roadmap:
# results/feature_importance_cocomo_rf.png, ..._cocomo_xgb.png,
# ..._desharnais_rf.png, ..._desharnais_xgb.png
#
# "Software-Unified"/"Construction" tags are also used by
# select_final_model.resave_model() for the multi-domain extension's
# final_model_{tag}.pkl filenames, even though those two domains'
# feature-importance charts are generated directly by
# src/train_new_domains.py (one winner-model chart each, not the
# per-algorithm rf/xgb pair analyze_model() below produces).
DATASET_FILE_TAGS = {
    "COCOMO-NASA": "cocomo",
    "Desharnais": "desharnais",
    "Software-Unified": "software",
    "Construction": "construction",
}
MODEL_FILE_TAGS = {"random_forest": "rf", "xgboost": "xgb"}


def get_feature_names(train_path: str, target_col: str) -> list[str]:
    """Feature names in the exact order the model was trained on —
    the train CSV's columns minus the target, same file train_models.py
    used to fit the model.
    """
    train_df = pd.read_csv(train_path, nrows=1)
    return [c for c in train_df.columns if c != target_col]


def get_importance_ranking(model, feature_names: list[str]) -> pd.DataFrame:
    """Ranked (highest first) feature -> importance table for a tree
    model. Both RandomForestRegressor and XGBRegressor expose the same
    .feature_importances_ array (aligned with the training feature order),
    so this one function covers both.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(ranking_df: pd.DataFrame, title: str, output_path: str, top_n: int = CHART_TOP_N) -> None:
    """Horizontal bar chart, most important feature at the top."""
    plot_df = ranking_df.head(top_n).iloc[::-1]  # reverse: barh draws bottom-to-top

    fig_height = max(4, 0.35 * len(plot_df) + 1)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_model(dataset_name: str, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    slug = DATASET_SLUGS[dataset_name]
    model_path = f"{MODELS_DIR}/{slug}_{model_name}.pkl"
    model = load_artifact(model_path)

    ranking = get_importance_ranking(model, feature_names)

    print(f"--- {dataset_name} / {MODEL_LABELS[model_name]} ---")
    print(ranking.to_string(index=False))
    print()

    chart_path = f"{RESULTS_DIR}/feature_importance_{DATASET_FILE_TAGS[dataset_name]}_{MODEL_FILE_TAGS[model_name]}.png"
    plot_importance(
        ranking,
        title=f"{dataset_name}: Feature importance ({MODEL_LABELS[model_name]})",
        output_path=chart_path,
    )
    print(f"Saved -> {chart_path}")
    print()

    return ranking


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("EstimateX — Step 8 feature importance")
    print()

    for cfg in DATASET_CONFIGS:
        feature_names = get_feature_names(cfg["train_path"], cfg["target_col"])
        for model_name in TREE_MODEL_NAMES:
            analyze_model(cfg["name"], model_name, feature_names)

    print("=" * 78)
    print("Scoping note")
    print("=" * 78)
    print(
        "Linear Regression and SVR are not covered here: Linear Regression's "
        "coefficients aren't directly comparable across differently-scaled/encoded "
        "features without extra normalization work, and sklearn's SVR (default RBF "
        "kernel) has no coefficient/importance concept at all. Explainability in "
        "EstimateX is therefore currently tied to whichever tree-based model "
        "(Random Forest or XGBoost) Step 9 selects as best — a deliberate scoping "
        "decision, not an oversight."
    )
