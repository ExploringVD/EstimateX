"""
EstimateX — interactive Flask web interface (rebuilt for the multi-domain
extension).

Takes real project details from a form, applies the EXACT preprocessing
fitted for whichever domain was selected (loaded from
models/final_scaler_*.pkl / models/final_encoder_*.pkl — never refit
here), runs it through the corresponding final model
(models/final_model_*.pkl), and renders a real prediction with real,
per-request feature-importance explainability.

=====================================================================
ASSUMPTIONS MADE HERE — flagged explicitly, so they can be adjusted:
=====================================================================

1. DOMAIN -> MODEL ROUTING. "Software/Web", "Mobile App", and
   "Enterprise/Other" all route to the SAME model, models/final_model_
   software.pkl (the unified COCOMO-NASA + Desharnais model from the
   multi-domain extension) — there is only one trained software model,
   not three. Splitting the dropdown into three labels is purely so a
   user describing their project picks something that feels natural,
   not because the underlying prediction differs by sub-domain; this is
   stated here so it's never mistaken for three separate models.
   "Construction" routes to models/final_model_construction.pkl.

2. SOFTWARE FORM FIELDS vs. THE UNIFIED MODEL'S ACTUAL FEATURES. The
   unified software model has exactly 4 features: team_experience,
   project_size_kloc, complexity, source_dataset (see
   src/unify_software.py). The form still shows Team Size and Required
   Reliability (per the UI spec), but NEITHER is a feature of the
   unified model:
     - Team Size: neither original dataset ever had a headcount column
       (same reason as before the multi-domain extension) — kept in the
       form, flagged as informational-only in its hint text.
     - Required Reliability: the unified schema deliberately kept only
       the 4 columns the roadmap specified (team size/experience,
       project size, complexity) — reliability wasn't part of that
       unified schema. Kept in the form per the UI spec, flagged as
       informational-only, same honest treatment as Team Size.
   Complexity and Team Experience map directly onto the model's own
   `complexity` and `team_experience` features. Project Size (KLOC) maps
   directly onto `project_size_kloc`.
   `source_dataset` has no real-world form equivalent — it's a
   training-data-provenance flag, not a property of a new hypothetical
   project. It's set to the MODE of the unified training data (computed
   at startup, see SOFTWARE_SOURCE_DEFAULT) — its feature importance is
   the lowest of the 4 (~0.005, see results/feature_importance_software.png),
   so this default has minimal effect on predictions either way.

3. CONSTRUCTION FORM FIELDS. All 8 of the construction model's real
   features (locality, floor area, lot area, the three preliminary-cost
   variants, planned duration, starting unit price — see
   results/construction_domain_notes.md) are exposed as form fields
   directly; none need to be defaulted, since none are "missing" from
   what a user would plausibly know before construction starts.

4. CONSTRUCTION PREDICTS COST ONLY, NOT TIME. The dataset's own creators
   only label two columns "(output)": sales price and construction
   cost. Planned duration is an INPUT the model uses, not something it
   was ever trained to predict — there's no "actual duration" outcome
   in the data to train or evaluate against. The results page therefore
   never shows a "Predicted Effort" stat for Construction, and shows an
   explicit note instead of fabricating one.

5. PREDICTED COST (software domains). Neither original software dataset
   has a currency column — only EFFORT. "Predicted Cost" for software is
   a DERIVED, illustrative figure: effort (in person-months) x an
   assumed COST_PER_PERSON_MONTH constant. Construction's cost, by
   contrast, is the model's actual predicted target (V-10, "10000 IRR"
   units per results/construction_domain_notes.md) — displayed with a
   "$" prefix to match the UI's shared "Predicted Cost" stat, which is
   an approximation across a currency/unit boundary worth being upfront
   about rather than silently implying it's USD.

6. "WHAT DROVE THIS ESTIMATE" is per-request, not just the model's
   global .feature_importances_ (which is the same ranking every time).
   Each feature's contribution here is approximated as
   importance x |this request's scaled value for that feature| — a
   cheap heuristic (not SHAP/LIME), documented as an approximation, not
   a formal per-instance attribution method.

7. THE NUMBERED FACTOR SENTENCES (Part C) pair each request's actual
   top-ranked features and percentages (real, computed per request) with
   a per-feature-TYPE explanatory clause (why that kind of factor
   generally moves an estimate — e.g. "larger codebases require more
   testing effort"). The qualitative reasoning template for a given
   feature doesn't change between requests; which features appear, in
   what order, and at what percentage does. Documented here so this
   isn't mistaken for a fully dynamic natural-language generator.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import (  # noqa: E402
    COCOMO_ORDINAL_SCALE,
    encode_onehot,
    encode_ordinal,
    load_artifact,
    scale_numeric_features,
)
from unify_software import (  # noqa: E402
    COMPLEXITY_COL,
    PROJECT_SIZE_COL,
    SCALE_COLS as SOFTWARE_SCALE_COLS,
    SOURCE_COL,
    TEAM_EXPERIENCE_COL,
)
from preprocess_construction import (  # noqa: E402
    NUMERIC_COLS as CONSTRUCTION_NUMERIC_COLS,
    ONEHOT_COLS as CONSTRUCTION_ONEHOT_COLS,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants / documented assumptions (see module docstring for the "why")
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

COST_PER_PERSON_MONTH = 10_000  # illustrative, software domains only — see assumption 5

TOP_N_FEATURES = 5

MAX_TEAM_SIZE = 2_000
MAX_PROJECT_SIZE_KLOC = 100_000

DOMAIN_OPTIONS = [
    ("software_web", "Software / Web"),
    ("mobile_app", "Mobile App"),
    ("enterprise_other", "Enterprise / Other"),
    ("construction", "Construction"),
]
DOMAIN_LABELS = dict(DOMAIN_OPTIONS)
SOFTWARE_DOMAIN_VALUES = {"software_web", "mobile_app", "enterprise_other"}
CONSTRUCTION_DOMAIN_VALUE = "construction"

ORDINAL_OPTIONS = [
    ("vl", "Very Low"), ("l", "Low"), ("n", "Nominal"),
    ("h", "High"), ("vh", "Very High"), ("xh", "Extra High"),
]
EXPERIENCE_OPTIONS = [
    ("0", "Novice (little to no experience)"),
    ("1", "Junior"),
    ("2", "Intermediate"),
    ("3", "Senior"),
    ("4", "Expert"),
]
EXPERIENCE_VALUES = {v for v, _ in EXPERIENCE_OPTIONS}
ORDINAL_VALUES = {v for v, _ in ORDINAL_OPTIONS}
LOCALITY_OPTIONS = list(range(1, 21))  # the dataset's own 20 zone codes
LOCALITY_VALUES = {str(v) for v in LOCALITY_OPTIONS}

DEFAULT_FORM_VALUES = {
    "domain": "software_web",
    "team_size": "8",
    "team_experience": "2",
    "project_size_kloc": "50",
    "complexity": "n",
    "reliability": "n",
    "locality": "1",
    "floor_area_m2": "1220",
    "lot_area_m2": "300",
    "prelim_est_cost_total": "165",
    "prelim_est_cost": "140",
    "prelim_est_cost_equiv": "522",
    "planned_duration": "6",
    "unit_price_start": "805",
}

# Friendlier labels for the feature-importance chart / factor list.
FEATURE_LABELS = {
    "team_experience": "Team Experience",
    "project_size_kloc": "Project Size (KLOC)",
    "complexity": "Project Complexity",
    "source_dataset": "Historical Data Source",
    "floor_area_m2": "Total Floor Area",
    "lot_area_m2": "Lot Area",
    "prelim_est_cost_total": "Preliminary Cost Estimate (Total)",
    "prelim_est_cost": "Preliminary Cost Estimate",
    "prelim_est_cost_equiv": "Preliminary Cost Estimate (Base-Year)",
    "planned_duration": "Planned Duration",
    "unit_price_start": "Starting Unit Price",
    "locality": "Locality Zone",
}

# Why each factor TYPE generally moves an estimate — see assumption 7.
FEATURE_EXPLANATIONS = {
    "team_experience": "team experience shapes how efficiently the work gets done, which pulls the estimate in one direction or the other",
    "project_size_kloc": "larger codebases require proportionally more implementation and testing effort, which scales the estimate up",
    "complexity": "higher complexity ratings mean more intricate logic and integration work, which increases the estimate",
    "source_dataset": "reflects which historical dataset this project's profile most resembles — a modeling artifact, not something you control",
    "floor_area_m2": "bigger buildings require more materials and labor, which scales the estimate up",
    "lot_area_m2": "larger lots are often associated with bigger or more complex site work",
    "prelim_est_cost_total": "the total preliminary estimate made before construction — historically a strong predictor of the final cost",
    "prelim_est_cost": "the preliminary cost estimate made before construction started — historically the single strongest predictor of the final actual cost",
    "prelim_est_cost_equiv": "the base-year-adjusted preliminary estimate, correcting for how prices shifted since the project started",
    "planned_duration": "longer planned timelines are associated with larger, more expensive projects",
    "unit_price_start": "the starting price per square meter reflects local material and labor costs at the time",
    "locality": "captures location-specific cost patterns from historical projects in this same zone",
}


def humanize_feature_name(raw_name: str) -> str:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    if "_" in raw_name:
        prefix, _, suffix = raw_name.partition("_")
        if prefix in FEATURE_LABELS:
            return f"{FEATURE_LABELS[prefix]}: {suffix}"
    return raw_name.replace("_", " ").title()


def explain_feature(raw_name: str) -> str:
    if raw_name in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[raw_name]
    if "_" in raw_name:
        prefix = raw_name.partition("_")[0]
        if prefix in FEATURE_EXPLANATIONS:
            return FEATURE_EXPLANATIONS[prefix]
    return "one of the factors this model weighs when producing an estimate"


# ---------------------------------------------------------------------------
# Load everything ONCE at startup: models + fitted preprocessing artifacts
# for both domains.
# ---------------------------------------------------------------------------

SOFTWARE_MODEL = load_artifact(f"{MODELS_DIR}/final_model_software.pkl")
SOFTWARE_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_software.pkl")
SOFTWARE_ENCODER = load_artifact(f"{MODELS_DIR}/final_encoder_software.pkl")  # OrdinalEncoder, complexity

CONSTRUCTION_MODEL = load_artifact(f"{MODELS_DIR}/final_model_construction.pkl")
CONSTRUCTION_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_construction.pkl")
CONSTRUCTION_ENCODER = load_artifact(f"{MODELS_DIR}/final_encoder_construction.pkl")  # OneHotEncoder, locality

# Mode of source_dataset in the unified training data — see assumption 2.
_unified_df = pd.read_csv(f"{PROCESSED_DIR}/software_unified.csv")
SOFTWARE_SOURCE_DEFAULT = int(_unified_df[SOURCE_COL].mode().iloc[0])


# ---------------------------------------------------------------------------
# Transform-only preprocessing (reuses src/preprocessing.py's exact
# functions with the loaded, already-fitted scaler/encoders).
# ---------------------------------------------------------------------------

def transform_software_row(raw_row_df: pd.DataFrame) -> pd.DataFrame:
    df, _ = encode_ordinal(raw_row_df, [COMPLEXITY_COL], COCOMO_ORDINAL_SCALE, encoder=SOFTWARE_ENCODER)
    df, _ = scale_numeric_features(df, SOFTWARE_SCALE_COLS, scaler=SOFTWARE_SCALER)
    return df[SOFTWARE_MODEL.feature_names_in_]


def transform_construction_row(raw_row_df: pd.DataFrame) -> pd.DataFrame:
    df, _ = encode_onehot(raw_row_df, CONSTRUCTION_ONEHOT_COLS, encoder=CONSTRUCTION_ENCODER)
    df, _ = scale_numeric_features(df, CONSTRUCTION_NUMERIC_COLS, scaler=CONSTRUCTION_SCALER)
    return df[CONSTRUCTION_MODEL.feature_names_in_]


def rank_features_for_request(model, transformed_row: pd.DataFrame) -> list[dict]:
    """'What drove THIS estimate' — see assumption 6: importance x |this
    row's scaled value|, not raw global importance.
    """
    importances = model.feature_importances_
    values = transformed_row.iloc[0].abs().to_numpy()
    salience = importances * values

    fi_df = pd.DataFrame({"feature": transformed_row.columns, "salience": salience})
    fi_df = fi_df.sort_values("salience", ascending=False).head(TOP_N_FEATURES).reset_index(drop=True)

    total = fi_df["salience"].sum()
    # 5-level opacity scale, top rank at full opacity.
    opacities = [1, 0.78, 0.6, 0.42, 0.28]
    rows = []
    for i, row in fi_df.iterrows():
        pct = float(row["salience"] / total * 100) if total > 0 else 0.0
        rows.append(
            {
                "raw_feature": row["feature"],
                "label": humanize_feature_name(row["feature"]),
                "pct": round(pct, 1),
                "opacity": opacities[i] if i < len(opacities) else opacities[-1],
            }
        )
    return rows


def build_factor_sentences(feature_rows: list[dict]) -> list[str]:
    """Numbered, plain-language list below the bar chart (Part C) — real
    per-request feature + percentage, paired with a per-feature-type
    explanatory clause (see assumption 7).
    """
    sentences = []
    for row in feature_rows:
        explanation = explain_feature(row["raw_feature"])
        sentences.append(f"{row['label']} ({row['pct']}%) — {explanation}.")
    return sentences


# ---------------------------------------------------------------------------
# Prediction pipelines
# ---------------------------------------------------------------------------

def predict_software(complexity: str, team_experience: str, kloc: float) -> dict:
    # The unified model's `team_experience` feature is a 0-5-ish average
    # (see src/unify_software.py — COCOMO's ordinal ratings average to
    # 0-5, Desharnais's TeamExp/ManagerExp average to 0-4). The dropdown's
    # 0-4 "experience level" index is used directly as that same numeric
    # scale, consistent with how the ordinal complexity scale is reused
    # elsewhere rather than converted through an intermediate label.
    row = {
        TEAM_EXPERIENCE_COL: float(int(team_experience)),
        PROJECT_SIZE_COL: kloc,
        COMPLEXITY_COL: complexity,
        SOURCE_COL: SOFTWARE_SOURCE_DEFAULT,
    }

    raw_row_df = pd.DataFrame([row])
    transformed = transform_software_row(raw_row_df)

    predicted_effort_months = max(float(SOFTWARE_MODEL.predict(transformed)[0]), 0.0)
    feature_rows = rank_features_for_request(SOFTWARE_MODEL, transformed)

    return {
        "predicted_effort_months": predicted_effort_months,
        "predicted_cost": predicted_effort_months * COST_PER_PERSON_MONTH,
        "feature_rows": feature_rows,
        "is_construction": False,
    }


def predict_construction(form) -> dict:
    row = {
        "locality": int(form["locality"]),
        "floor_area_m2": float(form["floor_area_m2"]),
        "lot_area_m2": float(form["lot_area_m2"]),
        "prelim_est_cost_total": float(form["prelim_est_cost_total"]),
        "prelim_est_cost": float(form["prelim_est_cost"]),
        "prelim_est_cost_equiv": float(form["prelim_est_cost_equiv"]),
        "planned_duration": float(form["planned_duration"]),
        "unit_price_start": float(form["unit_price_start"]),
    }
    raw_row_df = pd.DataFrame([row])
    transformed = transform_construction_row(raw_row_df)

    predicted_cost = max(float(CONSTRUCTION_MODEL.predict(transformed)[0]), 0.0)
    feature_rows = rank_features_for_request(CONSTRUCTION_MODEL, transformed)

    return {
        "predicted_effort_months": None,
        "predicted_cost": predicted_cost,
        "feature_rows": feature_rows,
        "is_construction": True,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_positive_number(raw_value, label: str, integer: bool = False, max_value: float | None = None) -> str | None:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return f"{label} is required."
    try:
        value = int(raw_value) if integer else float(raw_value)
    except ValueError:
        return f"{label} must be a {'whole number' if integer else 'number'}."
    if value <= 0:
        return f"{label} must be a positive number."
    if max_value is not None and value > max_value:
        return f"{label} must be {max_value:,.0f} or fewer."
    return None


def validate_form(form) -> dict:
    """Returns a dict of field -> error message. Empty dict means valid.
    Only validates the fields relevant to the submitted domain.
    """
    errors = {}
    domain = form.get("domain")
    if domain not in DOMAIN_LABELS:
        errors["domain"] = "Please choose a project domain."
        return errors

    if domain in SOFTWARE_DOMAIN_VALUES:
        if form.get("complexity") not in ORDINAL_VALUES:
            errors["complexity"] = "Please choose a complexity level."
        if form.get("reliability") not in ORDINAL_VALUES:
            errors["reliability"] = "Please choose a reliability level."
        if form.get("team_experience") not in EXPERIENCE_VALUES:
            errors["team_experience"] = "Please choose a team experience level."

        err = _validate_positive_number(form.get("team_size"), "Team size", integer=True, max_value=MAX_TEAM_SIZE)
        if err:
            errors["team_size"] = err
        err = _validate_positive_number(form.get("project_size_kloc"), "Project size", max_value=MAX_PROJECT_SIZE_KLOC)
        if err:
            errors["project_size_kloc"] = err

    else:  # construction
        if form.get("locality") not in LOCALITY_VALUES:
            errors["locality"] = "Please choose a locality zone."

        numeric_fields = [
            ("floor_area_m2", "Floor area"),
            ("lot_area_m2", "Lot area"),
            ("prelim_est_cost_total", "Preliminary estimated cost (total)"),
            ("prelim_est_cost", "Preliminary estimated cost"),
            ("prelim_est_cost_equiv", "Preliminary estimated cost (base-year)"),
            ("planned_duration", "Planned duration"),
            ("unit_price_start", "Starting unit price"),
        ]
        for field, label in numeric_fields:
            err = _validate_positive_number(form.get(field), label)
            if err:
                errors[field] = err

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def render_form(form_values, errors):
    return render_template(
        "form.html",
        form_values=form_values,
        errors=errors,
        domain_options=DOMAIN_OPTIONS,
        ordinal_options=ORDINAL_OPTIONS,
        experience_options=EXPERIENCE_OPTIONS,
        locality_options=LOCALITY_OPTIONS,
        max_team_size=MAX_TEAM_SIZE,
        max_project_size_kloc=MAX_PROJECT_SIZE_KLOC,
    )


@app.route("/", methods=["GET"])
def index():
    return render_form(DEFAULT_FORM_VALUES, {})


@app.route("/predict", methods=["POST"])
def predict():
    errors = validate_form(request.form)
    if errors:
        return render_form(request.form, errors), 400

    domain = request.form["domain"]

    if domain in SOFTWARE_DOMAIN_VALUES:
        result = predict_software(
            complexity=request.form["complexity"],
            team_experience=request.form["team_experience"],
            kloc=float(request.form["project_size_kloc"]),
        )
    else:
        result = predict_construction(request.form)

    factor_sentences = build_factor_sentences(result["feature_rows"])

    # Software's cost is a real USD estimate (effort x an assumed rate —
    # see assumption 5); Construction's is the model's native-unit output
    # (x10,000 IRR, see results/construction_domain_notes.md) — formatted
    # differently here rather than both wearing a "$" that would falsely
    # imply Construction's figure is US dollars too.
    if result["is_construction"]:
        predicted_cost_display = f"{result['predicted_cost']:,.0f}"
    else:
        predicted_cost_display = f"${result['predicted_cost']:,.0f}"

    return render_template(
        "results.html",
        domain_label=DOMAIN_LABELS[domain],
        is_construction=result["is_construction"],
        predicted_effort_months=(
            round(result["predicted_effort_months"], 1) if result["predicted_effort_months"] is not None else None
        ),
        predicted_cost_display=predicted_cost_display,
        cost_per_person_month=COST_PER_PERSON_MONTH,
        feature_rows=result["feature_rows"],
        factor_sentences=factor_sentences,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
