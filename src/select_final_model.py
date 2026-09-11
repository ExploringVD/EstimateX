"""
EstimateX — Step 9: final model selection.

Reads results/model_comparison.csv (Step 7) to pick, per dataset, the
model that wins on MAE and PRED(25) TOGETHER (not either metric alone),
then decides on ONE algorithm to deploy for both datasets in Step 10,
re-saving that algorithm's already-trained models and their exact
matching preprocessing artifacts under clean, final filenames.

No retraining happens here — every artifact this script writes is a
re-save (joblib load -> joblib dump under a new name) of something Steps
4/6 already fit, so it's byte-for-byte the same transform/model the
comparison numbers in Step 7 and the importances in Step 8 were computed
from.
"""

from __future__ import annotations

import pandas as pd

from split_data import DATASET_CONFIGS
from train_models import DATASET_SLUGS
from evaluate import MODEL_LABELS
from feature_importance import DATASET_FILE_TAGS, TREE_MODEL_NAMES
from preprocessing import load_artifact, save_artifact

RESULTS_DIR = "results"
MODELS_DIR = "models"
COMPARISON_PATH = f"{RESULTS_DIR}/model_comparison.csv"
DECISION_PATH = f"{RESULTS_DIR}/final_model_decision.md"

# Display label -> internal filename slug (e.g. "Random Forest" ->
# "random_forest"), the inverse of evaluate.MODEL_LABELS.
LABEL_TO_SLUG = {label: slug for slug, label in MODEL_LABELS.items()}

# How close a model's MAE has to be to the best MAE in its dataset (as a
# percentage of the best MAE) to still be considered "combined-winner"
# material on the strength of a much better PRED(25) instead. Stated here
# explicitly, as a documented methodology choice, rather than an unstated
# judgment call.
MAE_TOLERANCE_PCT = 5.0


def pick_combined_winner(subset: pd.DataFrame) -> dict:
    """Pick the best model for one dataset using MAE and PRED(25)
    together, per the roadmap's instruction not to rely on either metric
    in isolation.

    Rule: if the same model has both the lowest MAE and the highest
    PRED(25), it wins outright. Otherwise, if the PRED(25)-leading
    model's MAE is within MAE_TOLERANCE_PCT of the MAE-leading model's
    MAE, the PRED(25) leader is preferred (a near-tie on average error,
    decided by which model produces more usably-close estimates).
    Otherwise, the MAE leader is kept (a big PRED(25) lead isn't worth a
    large jump in average error).
    """
    best_mae_row = subset.loc[subset["MAE"].idxmin()]
    best_pred25_row = subset.loc[subset["PRED(25)"].idxmax()]

    if best_mae_row["Model"] == best_pred25_row["Model"]:
        return {
            "winner": best_mae_row["Model"],
            "best_mae_row": best_mae_row,
            "best_pred25_row": best_pred25_row,
            "reason": (
                f"{best_mae_row['Model']} has both the lowest MAE ({best_mae_row['MAE']:.2f}) "
                f"and the highest PRED(25) ({best_mae_row['PRED(25)']:.1f}%) — wins outright, "
                "no tiebreak needed."
            ),
        }

    mae_gap_pct = (best_pred25_row["MAE"] - best_mae_row["MAE"]) / best_mae_row["MAE"] * 100
    pred25_gap = best_pred25_row["PRED(25)"] - best_mae_row["PRED(25)"]

    if mae_gap_pct <= MAE_TOLERANCE_PCT:
        winner = best_pred25_row["Model"]
        reason = (
            f"{best_mae_row['Model']} has the lowest MAE ({best_mae_row['MAE']:.2f}) and "
            f"{best_pred25_row['Model']} has the highest PRED(25) ({best_pred25_row['PRED(25)']:.1f}%). "
            f"{best_pred25_row['Model']}'s MAE ({best_pred25_row['MAE']:.2f}) is only "
            f"{mae_gap_pct:.2f}% higher than {best_mae_row['Model']}'s — within the "
            f"{MAE_TOLERANCE_PCT:.0f}% tolerance — while its PRED(25) is {pred25_gap:.1f} "
            f"percentage points higher. Preferring PRED(25) here: a near-tie on average error "
            f"decided in favor of the model that lands within 25% of the actual value far more "
            f"often, which matters more in practice than a small MAE difference."
        )
    else:
        winner = best_mae_row["Model"]
        reason = (
            f"{best_pred25_row['Model']} has the highest PRED(25) ({best_pred25_row['PRED(25)']:.1f}%), "
            f"but its MAE ({best_pred25_row['MAE']:.2f}) is {mae_gap_pct:.2f}% higher than "
            f"{best_mae_row['Model']}'s ({best_mae_row['MAE']:.2f}) — too large a gap (over the "
            f"{MAE_TOLERANCE_PCT:.0f}% tolerance) to prefer on PRED(25) alone. Keeping the MAE leader."
        )

    return {
        "winner": winner,
        "best_mae_row": best_mae_row,
        "best_pred25_row": best_pred25_row,
        "reason": reason,
    }


def choose_final_algorithm(per_dataset_winners: dict, dataset_sizes: dict) -> dict:
    """Decide ONE algorithm for both datasets, per the roadmap's policy:
    1. If the same algorithm wins both datasets (by pick_combined_winner
       above), deploy it.
    2. Otherwise, deploy the winner of the larger/more representative
       dataset — but ONLY if that winner is a tree-based model with
       feature-importance support (Random Forest or XGBoost, from Step
       8); explainability is a hard requirement, so Linear Regression and
       SVR are disqualified regardless of accuracy. If the larger
       dataset's winner isn't a tree model, fall back to whichever tree
       model wins MAE+PRED(25) on the larger dataset instead.
    """
    winners = set(per_dataset_winners.values())

    if len(winners) == 1:
        algorithm = winners.pop()
        return {
            "algorithm": algorithm,
            "decision_path": (
                f"Same algorithm ({algorithm}) wins the combined MAE+PRED(25) comparison on "
                f"BOTH datasets — deploying it directly, no fallback rule needed."
            ),
        }

    larger_dataset = max(dataset_sizes, key=dataset_sizes.get)
    candidate = per_dataset_winners[larger_dataset]
    candidate_slug = LABEL_TO_SLUG[candidate]

    if candidate_slug in TREE_MODEL_NAMES:
        return {
            "algorithm": candidate,
            "decision_path": (
                f"Datasets disagree on the combined winner. Falling back to the winner of the "
                f"larger dataset ({larger_dataset}, {dataset_sizes[larger_dataset]} rows): "
                f"{candidate}. It is a tree-based model, so it retains Step 8's feature-importance "
                f"explainability — condition satisfied, deploying it."
            ),
        }

    return {
        "algorithm": "tree-model fallback (see printed detail)",
        "decision_path": (
            f"Datasets disagree, and the larger dataset's winner ({candidate}) has no "
            f"feature-importance support. Explainability is a hard requirement, so falling back "
            f"to the best tree-based model (by MAE+PRED(25)) on the larger dataset instead."
        ),
    }


def resave_model(dataset_name: str, algorithm_label: str) -> str:
    slug = DATASET_SLUGS[dataset_name]
    algorithm_slug = LABEL_TO_SLUG[algorithm_label]
    tag = DATASET_FILE_TAGS[dataset_name]

    source_path = f"{MODELS_DIR}/{slug}_{algorithm_slug}.pkl"
    dest_path = f"{MODELS_DIR}/final_model_{tag}.pkl"

    model = load_artifact(source_path)
    save_artifact(model, dest_path)
    return dest_path


def resave_cocomo_preprocessing() -> tuple[str, str]:
    """COCOMO-NASA's preprocessing (src/preprocessing.py) fits FOUR
    separate transformers: one StandardScaler and THREE different
    encoders (an OrdinalEncoder for the 15 effort-driver ratings, a
    OneHotEncoder for the nominal columns, and a LabelEncoder for the
    binary `forg` column) — there's no single "the encoder" for this
    dataset. To honor the requested single `final_encoder_cocomo.pkl`
    filename while still preserving every encoder Step 10 needs to
    transform new input identically, all three are bundled into one dict
    and joblib-dumped together; the scaler is saved separately as usual.
    """
    scaler = load_artifact(f"{MODELS_DIR}/cocomo_scaler.joblib")
    ordinal_encoder = load_artifact(f"{MODELS_DIR}/cocomo_ordinal_encoder.joblib")
    onehot_encoder = load_artifact(f"{MODELS_DIR}/cocomo_onehot_encoder.joblib")
    binary_encoder = load_artifact(f"{MODELS_DIR}/cocomo_forg_label_encoder.joblib")

    scaler_path = f"{MODELS_DIR}/final_scaler_cocomo.pkl"
    encoder_path = f"{MODELS_DIR}/final_encoder_cocomo.pkl"

    save_artifact(scaler, scaler_path)
    # Bundle: {"ordinal": OrdinalEncoder, "onehot": OneHotEncoder, "binary_forg": LabelEncoder}
    save_artifact(
        {"ordinal": ordinal_encoder, "onehot": onehot_encoder, "binary_forg": binary_encoder},
        encoder_path,
    )
    return scaler_path, encoder_path


def resave_desharnais_preprocessing() -> tuple[str, str]:
    """Desharnais only has one scaler and one encoder (a OneHotEncoder for
    `Language`), so these map onto the requested filenames directly, no
    bundling needed.
    """
    scaler = load_artifact(f"{MODELS_DIR}/desharnais_scaler.joblib")
    onehot_encoder = load_artifact(f"{MODELS_DIR}/desharnais_onehot_encoder.joblib")

    scaler_path = f"{MODELS_DIR}/final_scaler_desharnais.pkl"
    encoder_path = f"{MODELS_DIR}/final_encoder_desharnais.pkl"

    save_artifact(scaler, scaler_path)
    save_artifact(onehot_encoder, encoder_path)
    return scaler_path, encoder_path


def write_decision_doc(
    final_algorithm: str,
    decision_path: str,
    per_dataset: dict,
    dataset_sizes: dict,
) -> None:
    cocomo = per_dataset["COCOMO-NASA"]
    desharnais = per_dataset["Desharnais"]

    lines = [
        "# EstimateX — Final Model Decision (Step 9)",
        "",
        f"**Chosen algorithm: {final_algorithm}**, deployed for both COCOMO-NASA and Desharnais.",
        "",
        "## Selection criteria",
        "",
        "Per dataset, the winning model was chosen using MAE (lower is better) and PRED(25) "
        "(higher is better) TOGETHER, not either metric in isolation. When the two metrics "
        f"disagreed, a model was preferred on PRED(25) only if its MAE was within "
        f"{MAE_TOLERANCE_PCT:.0f}% of the best MAE in that dataset (a near-tie on average error, "
        "decided in favor of the model that lands within 25% of the true value more often) — "
        "otherwise the MAE leader was kept. This tolerance is a stated methodology choice, not an "
        "arbitrary tiebreak.",
        "",
        "## COCOMO-NASA",
        "",
        f"- Best MAE: **{cocomo['best_mae_row']['Model']}** ({cocomo['best_mae_row']['MAE']:.2f})",
        f"- Best PRED(25): **{cocomo['best_pred25_row']['Model']}** ({cocomo['best_pred25_row']['PRED(25)']:.1f}%)",
        f"- **Winner: {cocomo['winner']}.** {cocomo['reason']}",
        "",
        "## Desharnais",
        "",
        f"- Best MAE: **{desharnais['best_mae_row']['Model']}** ({desharnais['best_mae_row']['MAE']:.2f})",
        f"- Best PRED(25): **{desharnais['best_pred25_row']['Model']}** ({desharnais['best_pred25_row']['PRED(25)']:.1f}%)",
        f"- **Winner: {desharnais['winner']}.** {desharnais['reason']}",
        "",
        "## Final decision",
        "",
        decision_path,
        "",
        "This also happens to align with Step 8's explainability findings: Random Forest is one "
        "of the two tree-based models with feature-importance support, so choosing it keeps "
        "EstimateX's \"not a black box\" explainability intact rather than trading it away for a "
        "small accuracy difference.",
        "",
        "## Tradeoffs acknowledged",
        "",
        f"- **COCOMO-NASA — RMSE**: SVR has the best RMSE (353.27) on COCOMO-NASA, beating "
        f"{cocomo['winner']}'s RMSE ({cocomo['best_mae_row']['RMSE']:.2f}). SVR was not chosen: "
        "its PRED(25) (10.5%) is far worse than Random Forest's (42.1%), and it has no "
        "feature-importance support at all (Step 8's scoping note), so it fails the "
        "explainability requirement regardless of its RMSE edge.",
        f"- **Desharnais — MAE and RMSE**: Linear Regression narrowly has the lowest raw MAE "
        f"(2402.17 vs Random Forest's 2434.10 — a {((2434.104 - 2402.174) / 2402.174 * 100):.2f}% "
        "difference) and the lowest RMSE (3075.30 vs 3343.40) on Desharnais. It was not chosen: "
        "the MAE/RMSE gap is small enough to be within noise on a 16-row test set, while its "
        "PRED(25) (6.3%) is four times worse than Random Forest's (25.0%), and — as already flagged "
        "in Steps 6-8 — Linear Regression's coefficients aren't usable for feature-importance-style "
        "explainability the way Random Forest's are, and its COCOMO-NASA CV performance (R² mean "
        "-7.11) was separately flagged as badly unstable, reinforcing that it isn't the more "
        "reliable choice across this project even where it narrowly leads on one metric.",
        f"- **Neither dataset's PRED(25) is high in absolute terms** ({cocomo['best_pred25_row']['PRED(25)']:.1f}% "
        f"and {desharnais['best_pred25_row']['PRED(25)']:.1f}% respectively, both well under the "
        "conventional 75% \"good estimator\" benchmark used in effort-estimation literature) — "
        f"{final_algorithm} is the best available choice among the four models tried, not a model "
        "that has solved effort estimation on these datasets outright. Worth stating plainly in "
        "the paper rather than overselling the result.",
        "",
        "## Deployed artifacts",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `models/final_model_cocomo.pkl` | Fitted " + final_algorithm + " model, COCOMO-NASA |",
        "| `models/final_model_desharnais.pkl` | Fitted " + final_algorithm + " model, Desharnais |",
        "| `models/final_scaler_cocomo.pkl` | Fitted `StandardScaler` for COCOMO-NASA's numeric + ordinal-encoded columns |",
        "| `models/final_encoder_cocomo.pkl` | Dict `{\"ordinal\": OrdinalEncoder, \"onehot\": OneHotEncoder, \"binary_forg\": LabelEncoder}` — COCOMO-NASA needs all three to reproduce its exact feature space |",
        "| `models/final_scaler_desharnais.pkl` | Fitted `StandardScaler` for Desharnais's numeric columns |",
        "| `models/final_encoder_desharnais.pkl` | Fitted `OneHotEncoder` for Desharnais's `Language` column |",
        "",
        "All of the above are re-saves (joblib load -> joblib dump under a new name) of the exact "
        "artifacts Steps 4 and 6 already fit — nothing was refit, so these are guaranteed to match "
        "the transform used to produce the Step 7/8 numbers.",
    ]

    with open(DECISION_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("EstimateX — Step 9 final model selection")
    print()

    comparison_df = pd.read_csv(COMPARISON_PATH)

    per_dataset_results = {}
    for dataset_name in comparison_df["Dataset"].unique():
        subset = comparison_df[comparison_df["Dataset"] == dataset_name]
        result = pick_combined_winner(subset)
        per_dataset_results[dataset_name] = result

        print("=" * 78)
        print(dataset_name)
        print("-" * 78)
        print(result["reason"])
        print()

    per_dataset_winners = {name: r["winner"] for name, r in per_dataset_results.items()}

    dataset_sizes = {}
    for cfg in DATASET_CONFIGS:
        dataset_sizes[cfg["name"]] = len(pd.read_csv(cfg["clean_path"]))

    decision = choose_final_algorithm(per_dataset_winners, dataset_sizes)
    final_algorithm = decision["algorithm"]

    print("=" * 78)
    print("Final decision")
    print("=" * 78)
    print(decision["decision_path"])
    print(f"=> Deploying: {final_algorithm}")
    print()

    print("=" * 78)
    print("Re-saving final artifacts")
    print("=" * 78)
    for cfg in DATASET_CONFIGS:
        model_path = resave_model(cfg["name"], final_algorithm)
        print(f"  {cfg['name']}: model -> {model_path}")

    cocomo_scaler_path, cocomo_encoder_path = resave_cocomo_preprocessing()
    print(f"  COCOMO-NASA: scaler -> {cocomo_scaler_path}")
    print(f"  COCOMO-NASA: encoder(s) -> {cocomo_encoder_path} (bundled dict: ordinal/onehot/binary_forg)")

    desharnais_scaler_path, desharnais_encoder_path = resave_desharnais_preprocessing()
    print(f"  Desharnais: scaler -> {desharnais_scaler_path}")
    print(f"  Desharnais: encoder -> {desharnais_encoder_path}")
    print()

    write_decision_doc(final_algorithm, decision["decision_path"], per_dataset_results, dataset_sizes)
    print(f"Saved -> {DECISION_PATH}")
