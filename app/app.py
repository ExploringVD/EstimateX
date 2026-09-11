"""
EstimateX — Step 10: interactive Flask web interface.

Takes real project details from a form, applies the EXACT preprocessing
fitted in Step 4 (loaded from models/final_scaler_*.pkl and
models/final_encoder_*.pkl — never refit here), runs it through the
Random Forest model chosen in Step 9 (models/final_model_*.pkl), and
renders a real prediction with real feature-importance explainability.

=====================================================================
ASSUMPTIONS MADE HERE — flagged explicitly, as requested, so they can
be adjusted if wrong:
=====================================================================

1. DOMAIN -> DATASET MAPPING. The roadmap's "Project Domain" dropdown
   can't literally offer domains like "Mobile" or "Construction" — we
   only have two trained datasets. The dropdown instead offers the two
   domains those datasets actually represent:
     - "Aerospace / Embedded Systems (NASA-style)" -> COCOMO-NASA model
     - "Business / Information Systems"            -> Desharnais model
   This is the most honest option: it doesn't imply support for domains
   we have no data for.

2. NOT EVERY FORM FIELD MAPS ONTO EVERY DATASET. COCOMO-NASA and
   Desharnais have genuinely different schemas (established repeatedly
   since Step 2/4), so a single 6-field form can't drive every raw
   feature of both models. Per field:
     - Project Complexity -> COCOMO's `cplx` directly. Desharnais has no
       complexity-like column at all, so this field is IGNORED on the
       Desharnais path (documented, not silently misused).
     - Required Reliability -> COCOMO's `rely` directly. Same story:
       ignored on the Desharnais path, no analogous column exists.
     - Team Experience -> COCOMO: applied identically to all 5 of its
       experience/capability ordinal drivers (acap, aexp, pcap, vexp,
       lexp), since "team experience" as one UI concept doesn't
       distinguish analyst capability from language experience the way
       COCOMO's original 15-driver questionnaire did. Desharnais: applied
       directly to both TeamExp and ManagerExp (same 0-4 scale the real
       data uses, confirmed against data/raw/desharnais.csv).
     - Team Size -> NOT used by either trained model. Neither raw dataset
       has a literal headcount column (COCOMO's drivers are
       capability/experience RATINGS, not headcount; Desharnais's
       TeamExp/ManagerExp are experience-in-years, not headcount either).
       Kept in the form per the UI spec and clearly flagged as
       informational-only in the form's helper text, rather than forcing
       it into an invented, undocumented proxy.
     - Estimated Project Size (KLOC) -> COCOMO: maps directly onto
       `equivphyskloc` (exact same unit, no conversion). Desharnais has no
       lines-of-code column at all — it measures size in function points
       instead. To still let this one shared "size" input meaningfully
       drive the Desharnais prediction (rather than ignoring size, which
       Step 8 found to be the single biggest driver of Desharnais effort),
       the KLOC figure is converted to an approximate function-point count
       using Capers Jones' commonly-cited "backfire" rule of thumb (~100
       lines of code per function point for typical 3GL languages — see
       BACKFIRE_LOC_PER_FP below), then every Desharnais size-related
       column (Transactions, Entities, PointsAdjust, Envergure,
       PointsNonAjust) is scaled by the same ratio relative to that
       column's typical (median) value. This is a deliberate, named,
       documented approximation — not a measured conversion.
   Every unmapped column (for both datasets) is filled with that column's
   median (numeric) or mode (categorical) from the real training data,
   computed once at app startup — see compute_*_defaults() below.

3. PREDICTED COST. Neither dataset has a currency/cost column — both
   only have EFFORT (COCOMO in person-months, Desharnais in person-hours).
   "Predicted Cost" is therefore a DERIVED, illustrative figure:
   effort (converted to person-months) x an assumed COST_PER_PERSON_MONTH
   constant (see below). This is clearly labeled in the UI as an
   assumption-based figure, not a second thing the model predicts, and
   COST_PER_PERSON_MONTH is a single constant to change if a real rate is
   available.

4. "WHAT DROVE THIS ESTIMATE" is per-request, not just the model's global
   .feature_importances_. Random Forest's built-in importances are a
   GLOBAL property of the model (the same ranking every time), which
   doesn't really answer "what drove THIS estimate." Instead, each
   feature's contribution here is approximated as
   importance x |this request's scaled value for that feature| — a cheap
   heuristic (not SHAP/LIME) that surfaces features that are BOTH
   generally important AND unusually different from a typical project in
   this specific request. Documented as an approximation, not a formal
   per-instance attribution method.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import (  # noqa: E402
    COCOMO_BINARY_COL,
    COCOMO_ID_COL,
    COCOMO_NUMERIC_COLS,
    COCOMO_ONEHOT_COLS,
    COCOMO_ORDINAL_COLS,
    COCOMO_ORDINAL_SCALE,
    COCOMO_TARGET_COL,
    DESHARNAIS_ID_COL,
    DESHARNAIS_NUMERIC_COLS,
    DESHARNAIS_ONEHOT_COLS,
    DESHARNAIS_TARGET_COL,
    encode_label_binary,
    encode_onehot,
    encode_ordinal,
    handle_missing_values,
    load_artifact,
    load_raw_csv,
    scale_numeric_features,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants / documented assumptions (see module docstring for the "why")
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Boehm/COCOMO's standard person-hours-per-person-month figure, used only
# to convert Desharnais's effort (person-hours) into the same
# "person-months" unit COCOMO's predictions are already in, so both
# domains display on a consistent scale.
HOURS_PER_PERSON_MONTH = 152

# Illustrative-only cost-per-person-month assumption, since neither
# dataset has a currency column. Change this to a real loaded labor rate
# if one is available.
COST_PER_PERSON_MONTH = 10_000  # USD

# Capers Jones' commonly-cited "backfire" rule of thumb: roughly 100 lines
# of code per function point for typical 3rd-generation languages. Used
# only to give the shared KLOC input a grounded (if approximate) effect on
# Desharnais's function-point-based size features.
BACKFIRE_LOC_PER_FP = 100

TOP_N_FEATURES = 5

# Sanity-check upper bounds for free-text number inputs — generous enough
# to comfortably cover any real project (and the "unusually large" stress
# tests in Step 11's own test plan, up to a few thousand KLOC), but tight
# enough to catch obviously-mistyped input like team_size=999999 with a
# clear error instead of silently accepting it. Found missing during
# Step 11 end-to-end testing; team_size=999999 was sailing through with
# no feedback before this was added.
MAX_TEAM_SIZE = 2_000
MAX_PROJECT_SIZE_KLOC = 100_000

DOMAIN_OPTIONS = [
    ("cocomo", "Aerospace / Embedded Systems (NASA-style)"),
    ("desharnais", "Business / Information Systems"),
]
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

DEFAULT_FORM_VALUES = {
    "domain": "cocomo",
    "complexity": "n",
    "reliability": "n",
    "team_size": "8",
    "team_experience": "2",
    "project_size_kloc": "50",
}

# Friendlier labels for the feature-importance chart. Anything not listed
# here falls back to a generic prettifier (see humanize_feature_name).
FEATURE_LABELS = {
    "equivphyskloc": "Project Size (KLOC)",
    "time": "Runtime Performance Constraint",
    "rely": "Required Reliability",
    "cplx": "Project Complexity",
    "year": "Project Year",
    "acap": "Analyst Capability",
    "aexp": "Analyst Experience",
    "pcap": "Programmer Capability",
    "vexp": "Platform Experience",
    "lexp": "Language Experience",
    "data": "Database Size",
    "stor": "Storage Constraint",
    "virt": "Platform Volatility",
    "turn": "Turnaround Time",
    "modp": "Modern Practices Use",
    "tool": "Tool Use",
    "sced": "Schedule Constraint",
    "forg": "Development Organization Type",
    "center": "NASA Center",
    "mode": "Development Mode",
    "projectname": "Project Type Code",
    "cat2": "Project Category",
    "TeamExp": "Team Experience (years)",
    "ManagerExp": "Manager Experience (years)",
    "YearEnd": "Project End Year",
    "Length": "Project Duration (months)",
    "Transactions": "Transaction Count",
    "Entities": "Entity Count",
    "PointsAdjust": "Adjusted Function Points",
    "Envergure": "Project Scope",
    "PointsNonAjust": "Unadjusted Function Points",
    "Language": "Programming Language",
}


def humanize_feature_name(raw_name: str) -> str:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    if "_" in raw_name:
        prefix, _, suffix = raw_name.partition("_")
        if prefix in FEATURE_LABELS:
            pretty_suffix = f"Type {suffix}" if prefix == "Language" else suffix.replace("_", " ").title()
            return f"{FEATURE_LABELS[prefix]}: {pretty_suffix}"
    return raw_name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Load everything ONCE at startup: models, fitted preprocessing artifacts,
# and per-column "typical project" defaults for fields the form doesn't
# collect.
# ---------------------------------------------------------------------------

COCOMO_MODEL = load_artifact(f"{MODELS_DIR}/final_model_cocomo.pkl")
COCOMO_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_cocomo.pkl")
COCOMO_ENCODERS = load_artifact(f"{MODELS_DIR}/final_encoder_cocomo.pkl")  # dict: ordinal/onehot/binary_forg

DESHARNAIS_MODEL = load_artifact(f"{MODELS_DIR}/final_model_desharnais.pkl")
DESHARNAIS_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_desharnais.pkl")
DESHARNAIS_ENCODER = load_artifact(f"{MODELS_DIR}/final_encoder_desharnais.pkl")


def compute_cocomo_defaults() -> dict:
    """Typical-project defaults for every COCOMO-NASA column the form
    doesn't collect: median for numeric columns, mode for categorical —
    computed from the real (cleaned, same as Step 4) training data, not
    guessed.
    """
    raw_df = load_raw_csv(f"{RAW_DIR}/cocomo_nasa.csv")
    df = raw_df.drop(columns=[COCOMO_ID_COL, COCOMO_TARGET_COL])
    categorical_cols = COCOMO_ONEHOT_COLS + COCOMO_ORDINAL_COLS + [COCOMO_BINARY_COL]
    df, _ = handle_missing_values(df, numeric_cols=COCOMO_NUMERIC_COLS, categorical_cols=categorical_cols)

    defaults = {}
    for col in df.columns:
        if col in COCOMO_NUMERIC_COLS:
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode(dropna=True).iloc[0]
    return defaults


def compute_desharnais_defaults() -> dict:
    raw_df = load_raw_csv(f"{RAW_DIR}/desharnais.csv")
    df = raw_df.drop(columns=[DESHARNAIS_ID_COL, DESHARNAIS_TARGET_COL])
    df, _ = handle_missing_values(df, numeric_cols=DESHARNAIS_NUMERIC_COLS, categorical_cols=DESHARNAIS_ONEHOT_COLS)

    defaults = {}
    for col in df.columns:
        if col in DESHARNAIS_NUMERIC_COLS:
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode(dropna=True).iloc[0]
    return defaults


COCOMO_DEFAULTS = compute_cocomo_defaults()
DESHARNAIS_DEFAULTS = compute_desharnais_defaults()


# ---------------------------------------------------------------------------
# Transform-only preprocessing (reuses src/preprocessing.py's exact
# functions with the loaded, already-fitted scaler/encoders — nothing is
# refit here, so this is guaranteed to match training).
# ---------------------------------------------------------------------------

def transform_cocomo_row(raw_row_df: pd.DataFrame) -> pd.DataFrame:
    df, _ = encode_ordinal(raw_row_df, COCOMO_ORDINAL_COLS, COCOMO_ORDINAL_SCALE, encoder=COCOMO_ENCODERS["ordinal"])
    df, _ = encode_label_binary(df, COCOMO_BINARY_COL, encoder=COCOMO_ENCODERS["binary_forg"])
    df, _ = encode_onehot(df, COCOMO_ONEHOT_COLS, encoder=COCOMO_ENCODERS["onehot"])
    scale_cols = COCOMO_NUMERIC_COLS + COCOMO_ORDINAL_COLS
    df, _ = scale_numeric_features(df, scale_cols, scaler=COCOMO_SCALER)
    return df[COCOMO_MODEL.feature_names_in_]


def transform_desharnais_row(raw_row_df: pd.DataFrame) -> pd.DataFrame:
    df, _ = encode_onehot(raw_row_df, DESHARNAIS_ONEHOT_COLS, encoder=DESHARNAIS_ENCODER)
    df, _ = scale_numeric_features(df, DESHARNAIS_NUMERIC_COLS, scaler=DESHARNAIS_SCALER)
    return df[DESHARNAIS_MODEL.feature_names_in_]


def rank_features_for_request(model, transformed_row: pd.DataFrame) -> list[dict]:
    """'What drove THIS estimate' — see assumption #4 in the module
    docstring: importance x |this row's scaled value|, not raw global
    importance, so the ranking reflects this specific input.
    """
    importances = model.feature_importances_
    values = transformed_row.iloc[0].abs().to_numpy()
    salience = importances * values

    fi_df = pd.DataFrame({"feature": transformed_row.columns, "salience": salience})
    fi_df = fi_df.sort_values("salience", ascending=False).head(TOP_N_FEATURES).reset_index(drop=True)

    total = fi_df["salience"].sum()
    rows = []
    for i, row in fi_df.iterrows():
        pct = float(row["salience"] / total * 100) if total > 0 else 0.0
        rows.append(
            {
                "label": humanize_feature_name(row["feature"]),
                "pct": round(pct, 1),
                "opacity": round(max(1 - i * 0.15, 0.25), 2),
            }
        )
    return rows


def build_summary_sentence(feature_rows: list[dict]) -> str:
    if not feature_rows:
        return "No dominant factor stood out for this estimate."
    top = feature_rows[:2]
    if len(top) == 1:
        return f"This estimate was shaped mainly by {top[0]['label']}."
    return f"This estimate was shaped mainly by {top[0]['label']} and {top[1]['label']}."


# ---------------------------------------------------------------------------
# Prediction pipelines
# ---------------------------------------------------------------------------

def predict_cocomo(complexity: str, reliability: str, team_experience: str, kloc: float) -> dict:
    row = dict(COCOMO_DEFAULTS)
    row["cplx"] = complexity
    row["rely"] = reliability
    row["equivphyskloc"] = kloc

    experience_label = COCOMO_ORDINAL_SCALE[int(team_experience)]
    for col in ("acap", "aexp", "pcap", "vexp", "lexp"):
        row[col] = experience_label

    raw_row_df = pd.DataFrame([row])
    transformed = transform_cocomo_row(raw_row_df)

    predicted_effort_months = max(float(COCOMO_MODEL.predict(transformed)[0]), 0.0)
    feature_rows = rank_features_for_request(COCOMO_MODEL, transformed)

    return {
        "domain_label": dict(DOMAIN_OPTIONS)["cocomo"],
        "predicted_effort_months": predicted_effort_months,
        "feature_rows": feature_rows,
    }


def predict_desharnais(team_experience: str, kloc: float) -> dict:
    row = dict(DESHARNAIS_DEFAULTS)

    team_exp_years = int(team_experience)
    row["TeamExp"] = team_exp_years
    row["ManagerExp"] = team_exp_years

    approx_function_points = (kloc * 1000) / BACKFIRE_LOC_PER_FP
    baseline_points_adjust = DESHARNAIS_DEFAULTS["PointsAdjust"]
    size_multiplier = approx_function_points / baseline_points_adjust if baseline_points_adjust else 1.0
    for col in ("Transactions", "Entities", "PointsAdjust", "Envergure", "PointsNonAjust"):
        row[col] = DESHARNAIS_DEFAULTS[col] * size_multiplier

    raw_row_df = pd.DataFrame([row])
    transformed = transform_desharnais_row(raw_row_df)

    predicted_effort_hours = max(float(DESHARNAIS_MODEL.predict(transformed)[0]), 0.0)
    predicted_effort_months = predicted_effort_hours / HOURS_PER_PERSON_MONTH
    feature_rows = rank_features_for_request(DESHARNAIS_MODEL, transformed)

    return {
        "domain_label": dict(DOMAIN_OPTIONS)["desharnais"],
        "predicted_effort_months": predicted_effort_months,
        "feature_rows": feature_rows,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_form(form) -> dict:
    """Returns a dict of field -> error message. Empty dict means valid."""
    errors = {}

    if form.get("domain") not in dict(DOMAIN_OPTIONS):
        errors["domain"] = "Please choose a project domain."
    if form.get("complexity") not in ORDINAL_VALUES:
        errors["complexity"] = "Please choose a complexity level."
    if form.get("reliability") not in ORDINAL_VALUES:
        errors["reliability"] = "Please choose a reliability level."
    if form.get("team_experience") not in EXPERIENCE_VALUES:
        errors["team_experience"] = "Please choose a team experience level."

    team_size_raw = (form.get("team_size") or "").strip()
    if not team_size_raw:
        errors["team_size"] = "Team size is required."
    else:
        try:
            team_size_value = int(team_size_raw)
            if team_size_value <= 0:
                errors["team_size"] = "Team size must be a positive whole number."
            elif team_size_value > MAX_TEAM_SIZE:
                errors["team_size"] = f"Team size must be {MAX_TEAM_SIZE:,} or fewer."
        except ValueError:
            errors["team_size"] = "Team size must be a whole number."

    kloc_raw = (form.get("project_size_kloc") or "").strip()
    if not kloc_raw:
        errors["project_size_kloc"] = "Project size is required."
    else:
        try:
            kloc_value = float(kloc_raw)
            if kloc_value <= 0:
                errors["project_size_kloc"] = "Project size must be a positive number."
            elif kloc_value > MAX_PROJECT_SIZE_KLOC:
                errors["project_size_kloc"] = f"Project size must be {MAX_PROJECT_SIZE_KLOC:,} KLOC or fewer."
        except ValueError:
            errors["project_size_kloc"] = "Project size must be a number."

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
    kloc = float(request.form["project_size_kloc"])

    if domain == "cocomo":
        result = predict_cocomo(
            complexity=request.form["complexity"],
            reliability=request.form["reliability"],
            team_experience=request.form["team_experience"],
            kloc=kloc,
        )
    else:
        result = predict_desharnais(
            team_experience=request.form["team_experience"],
            kloc=kloc,
        )

    predicted_cost = result["predicted_effort_months"] * COST_PER_PERSON_MONTH
    summary_sentence = build_summary_sentence(result["feature_rows"])

    return render_template(
        "results.html",
        domain_label=result["domain_label"],
        predicted_effort_months=round(result["predicted_effort_months"], 1),
        predicted_cost=predicted_cost,
        cost_per_person_month=COST_PER_PERSON_MONTH,
        feature_rows=result["feature_rows"],
        summary_sentence=summary_sentence,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
