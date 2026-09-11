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
   src/unify_software.py). The form asks for EXACTLY the 3 that are
   user-meaningful (Team Experience, Project Size, Project Complexity) —
   every field shown genuinely feeds the model, full stop.
   Team Size and Required Reliability were investigated (re-checked, not
   assumed) and REMOVED from the form entirely, not kept as decorative
   unused inputs:
     - Team Size (headcount): neither raw dataset has one. COCOMO-NASA's
       people-related columns (acap/aexp/pcap/vexp/lexp) are capability
       RATINGS, not counts; Desharnais's (TeamExp/ManagerExp) are
       experience in YEARS, not counts. A derived count (e.g. Effort /
       Length) was considered and rejected — it would use the training
       TARGET to build a feature, a leakage violation, not a real input.
     - Required Reliability: no analogous column in either dataset
       (already established since Step 10; re-confirmed here).
   Full investigation and proof (re-ran the whole unification + training
   pipeline afterward — output came back byte-for-byte identical,
   confirming there was truly nothing to add) is in
   results/software_unification_notes.md.
   `source_dataset` has no real-world form equivalent either — it's a
   training-data-provenance flag, not a property of a new hypothetical
   project, so it isn't a form field at all. It's set to the MODE of the
   unified training data (computed at startup, see
   SOFTWARE_SOURCE_DEFAULT) — its feature importance is the lowest of
   the 4 (~0.005, see results/feature_importance_software.png), so this
   default has minimal effect on predictions either way.

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
   assumed COST_PER_PERSON_MONTH constant, in USD. Construction's cost,
   by contrast, is the model's actual predicted target (V-10, "10000
   IRR" units per results/construction_domain_notes.md) — genuinely NOT
   a dollar figure, so it is shown in its own native unit with an
   explicit label, not converted or given a "$" (see #8 below for why
   dual-currency display doesn't extend to Construction).

6. DUAL CURRENCY DISPLAY (₹ / $) — SOFTWARE DOMAINS ONLY. Software's
   predicted cost genuinely is a USD figure (assumption 5), so showing
   it in both USD and INR at a fixed, labeled rate (INR_PER_USD below)
   is a legitimate display-only conversion. Construction's predicted
   cost is NOT a USD figure to begin with — it's the source dataset's
   native Iranian Rial unit — so running it through a USD-to-INR rate
   would chain two unrelated currencies (IRR -> a guessed USD rate ->
   INR) through a conversion this project has no grounded rate for,
   compounding one approximation with another rather than adding real
   information. Construction's cost is therefore left in its existing,
   clearly-labeled native-unit display; only software's ₹/$ figures are
   dual-currency.

7. "WHAT DROVE THIS ESTIMATE" is per-request, not just the model's
   global .feature_importances_ (which is the same ranking every time).
   Each feature's contribution here is approximated as
   importance x |this request's scaled value for that feature| — a
   cheap heuristic (not SHAP/LIME), documented as an approximation, not
   a formal per-instance attribution method.

8. THE NUMBERED FACTOR SENTENCES pair each request's actual
   top-ranked features and percentages (real, computed per request) with
   a per-feature-TYPE explanatory clause (why that kind of factor
   generally moves an estimate — e.g. "larger codebases require more
   testing effort"). The qualitative reasoning template for a given
   feature doesn't change between requests; which features appear, in
   what order, and at what percentage does. Documented here so this
   isn't mistaken for a fully dynamic natural-language generator.

9. ESTIMATED COMPLETION DATE — SOFTWARE DOMAINS ONLY. Software's form
   collects a planned start month/year; the results page adds the
   predicted effort (rounded to the nearest whole month) to it and shows
   "Estimated completion: around <Month> <Year>" (see
   compute_estimated_completion). This is a derived, approximate figure
   labeled as such — it inherits whatever error the underlying effort
   prediction carries, it is not a second, independently-modeled
   prediction. Construction has no effort/duration prediction to begin
   with (assumption 4), so it gets neither the start-date inputs nor a
   completion-date line at all.
"""

from __future__ import annotations

import os
import sys
from datetime import date

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

# Fixed, labeled conversion rate — a static demo rate, not a live fetch,
# per the roadmap's explicit instruction. Shown alongside every INR figure
# so it's never mistaken for a real-time rate.
INR_PER_USD = 85

TOP_N_FEATURES = 5

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

# Planned start month/year -> estimated completion date (software domains
# only — see predict_software / compute_estimated_completion below).
# Construction never predicts effort/duration at all (assumption 4 in the
# module docstring), so it gets neither the inputs nor a completion date.
MONTH_OPTIONS = [
    ("1", "January"), ("2", "February"), ("3", "March"), ("4", "April"),
    ("5", "May"), ("6", "June"), ("7", "July"), ("8", "August"),
    ("9", "September"), ("10", "October"), ("11", "November"), ("12", "December"),
]
MONTH_VALUES = {v for v, _ in MONTH_OPTIONS}
MONTH_NAMES = dict(MONTH_OPTIONS)
CURRENT_YEAR = date.today().year
MIN_START_YEAR = 1990
MAX_START_YEAR = CURRENT_YEAR + 50

DEFAULT_FORM_VALUES = {
    "domain": "software_web",
    "team_experience": "2",
    "project_size_kloc": "50",
    "complexity": "n",
    "start_month": "1",
    "start_year": str(CURRENT_YEAR),
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
# Start date -> estimated completion date (software domains only)
# ---------------------------------------------------------------------------

def compute_estimated_completion(start_month: int, start_year: int, effort_months: float) -> tuple[str, int]:
    """Adds the predicted effort (rounded to the nearest whole month) to a
    planned start month/year, correctly handling month/year rollover.

    Example: start_month=1 (Jan), start_year=2026, effort_months=14.2
      -> rounded_months = 14
      -> total = (1 - 1) + 14 = 14
      -> completion_month_index = 14 % 12 = 2 (0-based) -> March
      -> completion_year = 2026 + 14 // 12 = 2026 + 1 = 2027
      -> ("March", 2027)

    This is explicitly an APPROXIMATION derived from the predicted effort
    (itself an estimate with real error margin, see assumption 7) — never
    presented as a guaranteed date, see results.html's caption.
    """
    rounded_months = round(effort_months)
    total_months = (start_month - 1) + rounded_months
    completion_month_index = total_months % 12  # 0-based
    completion_year = start_year + total_months // 12
    completion_month_name = MONTH_NAMES[str(completion_month_index + 1)]
    return completion_month_name, completion_year


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
        if form.get("team_experience") not in EXPERIENCE_VALUES:
            errors["team_experience"] = "Please choose a team experience level."

        err = _validate_positive_number(form.get("project_size_kloc"), "Project size", max_value=MAX_PROJECT_SIZE_KLOC)
        if err:
            errors["project_size_kloc"] = err

        if form.get("start_month") not in MONTH_VALUES:
            errors["start_month"] = "Please choose a planned start month."

        raw_year = (form.get("start_year") or "").strip()
        if not raw_year:
            errors["start_year"] = "Planned start year is required."
        else:
            try:
                year_val = int(raw_year)
            except ValueError:
                errors["start_year"] = "Planned start year must be a whole number."
            else:
                if year_val < MIN_START_YEAR or year_val > MAX_START_YEAR:
                    errors["start_year"] = f"Planned start year must be between {MIN_START_YEAR} and {MAX_START_YEAR}."

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
        month_options=MONTH_OPTIONS,
        current_year=CURRENT_YEAR,
        min_start_year=MIN_START_YEAR,
        max_start_year=MAX_START_YEAR,
        locality_options=LOCALITY_OPTIONS,
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

    # Estimated completion date — software domains only (assumption 4:
    # Construction never predicts effort/duration, so there's nothing to
    # add a start date to). Derived from the SAME predicted effort shown
    # above, rounded to the nearest whole month (see
    # compute_estimated_completion's docstring for the rollover math).
    estimated_completion = None
    if not result["is_construction"]:
        completion_month_name, completion_year = compute_estimated_completion(
            start_month=int(request.form["start_month"]),
            start_year=int(request.form["start_year"]),
            effort_months=result["predicted_effort_months"],
        )
        estimated_completion = f"{completion_month_name} {completion_year}"

    # Software's cost is a real USD estimate (effort x an assumed rate —
    # see assumption 5) — shown in both USD and INR (assumption 6) at a
    # fixed, labeled rate. Construction's cost is NOT a dollar figure at
    # all (it's the source dataset's native Iranian Rial unit), so it
    # keeps its own clearly-labeled native-unit display rather than being
    # forced through an unrelated currency conversion (assumption 6).
    if result["is_construction"]:
        predicted_cost_usd = None
        predicted_cost_inr = None
        predicted_cost_native = round(result["predicted_cost"], 0)
    else:
        predicted_cost_usd = round(result["predicted_cost"], 0)
        predicted_cost_inr = round(result["predicted_cost"] * INR_PER_USD, 0)
        predicted_cost_native = None

    return render_template(
        "results.html",
        domain_label=DOMAIN_LABELS[domain],
        is_construction=result["is_construction"],
        predicted_effort_months=(
            round(result["predicted_effort_months"], 1) if result["predicted_effort_months"] is not None else None
        ),
        predicted_cost_usd=predicted_cost_usd,
        predicted_cost_inr=predicted_cost_inr,
        predicted_cost_native=predicted_cost_native,
        estimated_completion=estimated_completion,
        inr_per_usd=INR_PER_USD,
        cost_per_person_month=COST_PER_PERSON_MONTH,
        feature_rows=result["feature_rows"],
        factor_sentences=factor_sentences,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
