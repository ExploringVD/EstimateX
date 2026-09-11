"""
EstimateX — multi-domain extension: train, evaluate, and select the final
model for the two new domains (Software-Unified, Construction).

This is orchestration only — every actual step (splitting, cross-
validated training, test-set evaluation, model selection, artifact
re-saving, feature-importance charting) calls the EXACT SAME functions
already built for COCOMO-NASA/Desharnais in Steps 5-9
(split_data.split_and_save_dataset, train_models.train_and_save_dataset,
evaluate.evaluate_dataset, select_final_model.pick_combined_winner /
resave_model, feature_importance.get_feature_names /
get_importance_ranking / plot_importance). Nothing here reimplements
training, evaluation, or selection logic.
"""

from __future__ import annotations

import os

import pandas as pd

from split_data import split_and_save_dataset
from train_models import DATASET_SLUGS, train_and_save_dataset
from evaluate import evaluate_dataset
from feature_importance import (
    MODEL_LABELS,
    TREE_MODEL_NAMES,
    get_feature_names,
    get_importance_ranking,
    plot_importance,
)
from select_final_model import LABEL_TO_SLUG, pick_combined_winner, resave_model
from preprocessing import load_artifact, save_artifact

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
COMPARISON_PATH = f"{RESULTS_DIR}/model_comparison_new_domains.csv"

# One entry per new domain. `tag` matches DATASET_SLUGS/DATASET_FILE_TAGS
# (already registered in train_models.py / feature_importance.py) so the
# reused functions resolve save paths exactly as they do for the
# original two datasets.
NEW_DOMAIN_CONFIGS = [
    {
        "name": "Software-Unified",
        "tag": "software",
        "clean_path": f"{PROCESSED_DIR}/software_unified.csv",
        "target_col": "effort_person_months",
        "train_path": f"{PROCESSED_DIR}/software_unified_train.csv",
        "test_path": f"{PROCESSED_DIR}/software_unified_test.csv",
        "scaler_src": f"{MODELS_DIR}/software_unified_scaler.joblib",
        "encoder_src": f"{MODELS_DIR}/software_unified_ordinal_encoder.joblib",
    },
    {
        "name": "Construction",
        "tag": "construction",
        "clean_path": f"{PROCESSED_DIR}/construction_clean.csv",
        "target_col": "actual_construction_cost",
        "train_path": f"{PROCESSED_DIR}/construction_train.csv",
        "test_path": f"{PROCESSED_DIR}/construction_test.csv",
        "scaler_src": f"{MODELS_DIR}/construction_scaler.joblib",
        "encoder_src": f"{MODELS_DIR}/construction_onehot_encoder.joblib",
    },
]


def pick_explainable_winner(subset_df: pd.DataFrame) -> dict:
    """pick_combined_winner() alone only weighs MAE+PRED(25) — it doesn't
    know explainability is also a hard requirement. This applies Step 9's
    exact fallback rule (choose_final_algorithm: "if the winner isn't
    tree-based, fall back to whichever tree model wins on this dataset
    instead, since explainability is a hard requirement"), now per-domain
    rather than cross-dataset, since each new domain gets its own
    independent model choice instead of being reconciled against another
    dataset's winner.
    """
    result = pick_combined_winner(subset_df)
    winner_slug = LABEL_TO_SLUG[result["winner"]]
    if winner_slug in TREE_MODEL_NAMES:
        return result

    tree_only = subset_df[subset_df["Model"].map(lambda m: LABEL_TO_SLUG[m] in TREE_MODEL_NAMES)]
    fallback = pick_combined_winner(tree_only)
    fallback["reason"] = (
        f"By MAE+PRED(25) alone, {result['winner']} would win ({result['reason']}) — but it has no "
        f"feature-importance support, and explainability is a hard requirement (Step 9's rule). "
        f"Falling back to the best tree-based model instead: {fallback['reason']}"
    )
    return fallback


def resave_preprocessing(tag: str, scaler_src: str, encoder_src: str) -> tuple[str, str]:
    """Re-save the intermediate scaler/encoder under the final_* naming
    convention already established in Step 9 — same load-then-dump
    pattern as select_final_model.resave_cocomo_preprocessing /
    resave_desharnais_preprocessing, just parameterized instead of
    duplicated per domain, since (unlike COCOMO-NASA) neither new domain
    needs more than one encoder bundled together.
    """
    scaler = load_artifact(scaler_src)
    encoder = load_artifact(encoder_src)

    scaler_path = f"{MODELS_DIR}/final_scaler_{tag}.pkl"
    encoder_path = f"{MODELS_DIR}/final_encoder_{tag}.pkl"
    save_artifact(scaler, scaler_path)
    save_artifact(encoder, encoder_path)
    return scaler_path, encoder_path


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("EstimateX — multi-domain extension: training new domains")
    print("(reusing split_data / train_models / evaluate / select_final_model as-is)")
    print()

    all_comparison_rows = []

    for cfg in NEW_DOMAIN_CONFIGS:
        print("=" * 78)
        print(cfg["name"])
        print("=" * 78)

        split_result = split_and_save_dataset(
            name=cfg["name"],
            clean_path=cfg["clean_path"],
            target_col=cfg["target_col"],
            train_path=cfg["train_path"],
            test_path=cfg["test_path"],
        )
        print(f"Split: X_train={split_result['X_train_shape']} X_test={split_result['X_test_shape']}")

        cv_rows = train_and_save_dataset(
            name=cfg["name"],
            train_path=cfg["train_path"],
            test_path=cfg["test_path"],
            target_col=cfg["target_col"],
        )
        print(f"Trained {len(cv_rows)} models (4 algorithms, 5-fold CV each).")

        eval_rows = evaluate_dataset(name=cfg["name"], test_path=cfg["test_path"], target_col=cfg["target_col"])
        for row in eval_rows:
            print(f"  {row['Model']:20s} MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}  "
                  f"MMRE={row['MMRE']:.3f}  PRED(25)={row['PRED(25)']:.1f}%")
        all_comparison_rows.extend(eval_rows)

        subset_df = pd.DataFrame(eval_rows)
        winner_info = pick_explainable_winner(subset_df)
        print(f"Winner: {winner_info['winner']}")
        print(f"  {winner_info['reason']}")

        model_path = resave_model(cfg["name"], winner_info["winner"])
        scaler_path, encoder_path = resave_preprocessing(cfg["tag"], cfg["scaler_src"], cfg["encoder_src"])
        print(f"Saved -> {model_path}")
        print(f"Saved -> {scaler_path}")
        print(f"Saved -> {encoder_path}")

        # Feature-importance chart for the winning model, same charting
        # functions Step 8 used, one winner-model chart per the roadmap's
        # exact requested filename (results/feature_importance_construction.png).
        winner_slug = LABEL_TO_SLUG[winner_info["winner"]]
        slug = DATASET_SLUGS[cfg["name"]]
        model = load_artifact(f"{MODELS_DIR}/{slug}_{winner_slug}.pkl")
        feature_names = get_feature_names(cfg["train_path"], cfg["target_col"])
        ranking = get_importance_ranking(model, feature_names)
        chart_path = f"{RESULTS_DIR}/feature_importance_{cfg['tag']}.png"
        plot_importance(
            ranking,
            title=f"{cfg['name']}: Feature importance ({MODEL_LABELS[winner_slug]})",
            output_path=chart_path,
        )
        print(f"Saved -> {chart_path}")
        print()

    comparison_df = pd.DataFrame(all_comparison_rows).round(3)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    print(f"Saved -> {COMPARISON_PATH}")
    print()
    print(comparison_df.to_string(index=False))
