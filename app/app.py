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
   software.pkl (the unified COCOMO-NASA + Desharnais + China model —
   see results/software_unification_notes.md) — there is only one
   trained software model, not three. Splitting the dropdown into three
   labels is purely so a
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

5. PREDICTED COST (software domains). None of the three training
   datasets (COCOMO-NASA, Desharnais, China) has a currency column —
   only EFFORT. "Predicted Cost" for software is a DERIVED, illustrative
   figure: effort (in person-months) x an assumed $/person-month rate,
   in USD. That rate is now REGION-SELECTABLE (COST_REGIONS below,
   default "United States") rather than a single constant — see
   results/cost_regions_notes.md for why this is a safe business-rate
   lookup change rather than a machine-learning change (the effort
   prediction itself never changes with region, only the multiplier
   applied to it afterward). Construction's cost, by contrast, is the
   model's actual predicted target (V-10, "10000 IRR" units per
   results/construction_domain_notes.md) — genuinely NOT a dollar
   figure, so it is shown in its own native unit with an explicit label,
   not converted or given a "$", and does NOT get a region selector
   either (same file, "why Construction doesn't get this" — mixing an
   Iranian-Rial-denominated, already-locality-aware ML prediction with a
   generic software-salary figure would have no honest basis; see #8
   below for why dual-currency display doesn't extend to Construction).

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

10. REGION-SELECTABLE COST RATE — SOFTWARE DOMAINS ONLY. See
    results/cost_regions_notes.md for the full writeup (all 6 rates,
    sources, the salary-vs-vendor-rate methodology caveat, and the "why
    Construction doesn't get this" reasoning in full). Short version:
    software's $/person-month rate (assumption 5) is a business-rate
    lookup, not a trained model behavior — COST_REGIONS below holds 6
    representative markets, the form's "Cost basis region" dropdown
    picks one (default "United States", preserving prior behavior), and
    predict_software() multiplies the SAME effort prediction by whichever
    rate was picked. Nothing about the trained model, its features, or
    its effort prediction changes with region. Construction is
    deliberately excluded — its cost is a real, currency-denominated,
    already-locality-aware ML prediction (see assumption 5), not an
    effort-times-rate estimate, so a generic regional salary figure has
    no honest relationship to it.
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
    TARGET_COL as SOFTWARE_TARGET_COL,
    TEAM_EXPERIENCE_COL,
    build_unified_dataframe,
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

# Region-selectable $/person-month rate — SOFTWARE DOMAINS ONLY. See
# assumption 5: none of the three training datasets (COCOMO-NASA,
# Desharnais, China) contain any real cost/salary data — the model only
# ever learned EFFORT. The $/month multiplier has always been an external
# business assumption bolted on afterward, which is exactly why it's safe
# to make it region-selectable: this is a business-rate lookup change, not
# a machine-learning change. Effort prediction itself is completely
# unaffected by region.
#
# Six representative markets, (value, label, $/person-month). Sourced and
# spot-checked 2026-09-12 — see results/cost_regions_notes.md for the full
# writeup, every source, and the explicit caveat below.
#   - us: $9,300/mo — ZipRecruiter Sept 2026 US national avg software
#     developer salary ($111,845/yr / 12 = $9,320/mo, rounded to $9,300).
#     SPOT-CHECKED directly against ziprecruiter.com/Salaries/Software-
#     Developer-Salary — the $111,845/yr figure is real and current.
#   - uk: $6,000/mo — GBP56,914/yr avg salary (Payprecision 2026) ~
#     GBP4,743/mo, converted at a STATIC approximate rate of GBP1=$1.27
#     (not live, same spirit as INR_PER_USD below).
#   - w_europe: $5,800/mo — EUR64,700/yr avg salary (jobvector 2026, via
#     WBS Coding School) ~ EUR5,392/mo, converted at a static approximate
#     EUR1=$1.08.
#   - e_europe: $3,600/mo — EUR35-45K/yr national median (EuroTopTech
#     2026), midpoint EUR40K/yr ~ EUR3,333/mo, same static conversion.
#   - india: $4,500/mo — blended mid-level offshore/vendor rate, midpoint
#     of $3,500-5,500/mo (Acquaintsoft 2026 rate card). SPOT-CHECKED
#     against acquaintsoft.com's 2026 rate card: real offshore hourly
#     rates there run $20-40/hr generally ($28-38/hr for a vetted
#     mid-level hire specifically) — at ~160 hrs/month that's roughly
#     $3,200-6,400/mo, the same order of magnitude as the $3,500-5,500/mo
#     claimed here, though the exact midpoint isn't independently
#     reproducible to the dollar from a single hourly figure.
#   - sea: $4,000/mo — blended vendor rate, midpoint of $3,440-4,900/mo
#     (Linnoedge 2026), not independently re-verified.
#
# METHODOLOGY CAVEAT (stated plainly, not smoothed over): US/UK/Germany/
# Poland are raw AVERAGE SALARIES; India/Vietnam are blended
# OFFSHORE-VENDOR RATES — that's what's actually well-documented for those
# markets, but it's not one uniform methodology end to end. That's also
# why India and Poland land close together ($4,500 vs $3,600) despite the
# usual "Eastern Europe costs more than India" framing — a vendor rate and
# a raw salary average aren't measuring quite the same thing. Don't read
# adjacent regions in this table as precisely ranked; the real signal is
# the ~2.6x spread between the cheapest (Poland, $3,600) and most
# expensive (US, $9,300) markets, not fine-grained ordering between
# neighbors.
COST_REGIONS = [
    ("us", "United States", 9300),
    ("uk", "United Kingdom", 6000),
    ("w_europe", "Western Europe (Germany)", 5800),
    ("e_europe", "Eastern Europe (Poland)", 3600),
    ("india", "India", 4500),
    ("sea", "Southeast Asia (Vietnam)", 4000),
]
COST_REGION_LABELS = {v: label for v, label, _ in COST_REGIONS}
COST_REGION_RATES = {v: rate for v, _, rate in COST_REGIONS}
COST_REGION_VALUES = set(COST_REGION_RATES)
COST_REGION_OPTIONS = [(v, label) for v, label, _ in COST_REGIONS]  # (value, label) pairs for the form dropdown
DEFAULT_COST_REGION = "us"  # preserves prior behavior for anyone who doesn't touch the field

# Fixed, labeled conversion rate — a static demo rate, not a live fetch,
# per the roadmap's explicit instruction. Shown alongside every INR figure
# so it's never mistaken for a real-time rate.
INR_PER_USD = 85

TOP_N_FEATURES = 5

MAX_PROJECT_SIZE_KLOC = 100_000

# Quick-fill presets for the Project Size (KLOC) field — SOFTWARE DOMAINS
# ONLY (Construction has no KLOC field at all). Pure UI convenience: these
# are NOT a new model input or categorical feature — clicking a preset
# just fills the same numeric project_size_kloc field a user would
# otherwise type into by hand; the backend never sees anything but that
# one existing field, exactly as before.
#
# Values are the 25th / 50th / 75th percentile of project_size_kloc
# across the full combined COCOMO-NASA + Desharnais + China unified
# training data (see src/unify_software.build_unified_dataframe()) —
# data-derived, not round-number guesses. Verified directly against a
# live rebuild of that data (2026-09-12): actual percentiles came back
# 11.5 / 23.4 / 48.7 KLOC over 669 rows (vs. this feature's originally
# supplied 12 / 24 / 50 over a claimed 671 rows) — close enough (within
# ~4%) that the small discrepancy is almost certainly just which exact
# pipeline rerun/imputation draw the percentiles were read from, not a
# different definition. The clean round numbers below are kept as the
# user-facing presets (a "Small" button reading 11.5 KLOC would look like
# a typo, not a deliberate choice) while this comment keeps the verified
# real numbers on record.
KLOC_PRESETS = [
    ("small", "Small", 12),
    ("medium", "Medium", 24),
    ("large", "Large", 50),
]

DOMAIN_OPTIONS = [
    ("software_web", "Software / Web"),
    ("mobile_app", "Mobile App"),
    ("enterprise_other", "Enterprise / Other"),
    ("construction", "Construction"),
]
DOMAIN_LABELS = dict(DOMAIN_OPTIONS)
SOFTWARE_DOMAIN_VALUES = {"software_web", "mobile_app", "enterprise_other"}

ORDINAL_OPTIONS = [
    ("vl", "Very Low"), ("l", "Low"), ("n", "Nominal"),
    ("h", "High"), ("vh", "Very High"), ("xh", "Extra High"),
]
ORDINAL_LABELS = dict(ORDINAL_OPTIONS)
EXPERIENCE_OPTIONS = [
    ("0", "Novice (little to no experience)"),
    ("1", "Junior"),
    ("2", "Intermediate"),
    ("3", "Senior"),
    ("4", "Expert"),
]
EXPERIENCE_VALUES = {v for v, _ in EXPERIENCE_OPTIONS}
EXPERIENCE_LABELS = dict(EXPERIENCE_OPTIONS)
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
    "cost_region": DEFAULT_COST_REGION,
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

# Features whose relationship to the estimate has a clear, already-
# documented sign in FEATURE_EXPLANATIONS above ("higher X increases/
# scales up the estimate"). For these, build_factor_sentences() below
# states whether THIS request's value is above or below what's typical in
# the training data, instead of always saying "higher X increases the
# estimate" regardless of whether the actual value is low or high — the
# bug this fixes: a Low-complexity request still read "higher complexity
# ratings... increases the estimate," which contradicts the Low pick.
#
# team_experience, source_dataset, and locality are deliberately NOT
# here: team_experience's relationship is genuinely two-directional (more
# OR less experience can be the efficient direction depending on the
# project — there's no single "more is worse/better" sign to state, which
# is why its existing wording already says "pulls the estimate in one
# direction or the other" rather than claiming a fixed direction), and
# source_dataset/locality are categorical/provenance flags with no
# meaningful "typical value" to be above or below. Their existing neutral
# phrasing is correct and is left untouched.
DIRECTIONAL_FEATURES = {
    "complexity", "project_size_kloc",
    "floor_area_m2", "lot_area_m2", "prelim_est_cost_total", "prelim_est_cost",
    "prelim_est_cost_equiv", "unit_price_start", "planned_duration",
}

# Magnitude bands on the SAME z-score already computed for salience in
# rank_features_for_request (not a separate calculation) — how strongly to
# word the direction. Verified against live data before picking these
# cutoffs: Low complexity (the bug report's actual case) has |z|~2.66
# ("well below"), while High complexity's |z|~0.15 now lands in the
# "close to typical" band since China made High the modal category — an
# honest reflection of the real training distribution, not a bug.
DIRECTIONAL_NEAR_TYPICAL_Z = 0.2
DIRECTIONAL_STRONG_Z = 1.0


def _format_directional_value(raw_feature: str, raw_value) -> str:
    """Human-readable description of THIS request's raw (pre-scaled)
    value for a DIRECTIONAL_FEATURES feature — reuses the same label
    dicts (ORDINAL_LABELS) and unit conventions already used elsewhere in
    this app (form.html's field labels/hints) rather than inventing new
    ones.
    """
    if raw_feature == COMPLEXITY_COL:
        return f"{ORDINAL_LABELS.get(raw_value, raw_value)} complexity rating"
    if raw_feature == PROJECT_SIZE_COL:
        return f"{raw_value:g} KLOC size"
    if raw_feature == "floor_area_m2":
        return f"{raw_value:g} m² floor area"
    if raw_feature == "lot_area_m2":
        return f"{raw_value:g} m² lot area"
    if raw_feature == "prelim_est_cost_total":
        return f"preliminary cost estimate (total) of {raw_value:g} (×10,000,000 IRR)"
    if raw_feature == "prelim_est_cost":
        return f"preliminary cost estimate of {raw_value:g} (×10,000 IRR)"
    if raw_feature == "prelim_est_cost_equiv":
        return f"base-year preliminary cost estimate of {raw_value:g} (×10,000 IRR)"
    if raw_feature == "unit_price_start":
        return f"starting unit price of {raw_value:g} (×10,000 IRR per m²)"
    if raw_feature == "planned_duration":
        return f"planned duration of {raw_value:g}"
    return f"{raw_value}"


def _direction_sentence(label: str, pct: float, value_phrase: str, z: float) -> str:
    """The direction-aware replacement for a DIRECTIONAL_FEATURES factor
    sentence — uses the SIGN (and magnitude) of the request's own scaled
    value, not just its salience-ranking magnitude, to say whether this
    value is pulling the estimate down or pushing it up.
    """
    if abs(z) < DIRECTIONAL_NEAR_TYPICAL_Z:
        return (
            f"{label} ({pct}%) — this project's {value_phrase} is close to what's typical "
            f"in the training data, so it isn't pulling the estimate strongly in either direction."
        )
    intensity = "well " if abs(z) >= DIRECTIONAL_STRONG_Z else ""
    comparison = "below" if z < 0 else "above"
    verb = "pulling" if z < 0 else "pushing"
    direction = "down" if z < 0 else "up"
    return (
        f"{label} ({pct}%) — this project's {value_phrase} is {intensity}{comparison} what's "
        f"typical in the training data, which is {verb} this estimate {direction}."
    )


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
# Training-data coverage per complexity level (RAW, unscaled units) — lets
# predict_software() honestly flag a request that extrapolates beyond any
# real historical example the model was actually trained on, instead of
# presenting a number with false confidence. See
# results/software_unification_notes.md: originally (COCOMO-NASA +
# Desharnais only) the smallest real Nominal-complexity project on record
# was 10 KLOC / 48 person-months, so a 1-KLOC "Nominal" query had ZERO
# comparable real training row — documented as an "Honest residual
# limitation — not fixable by retraining" on those two datasets alone.
# Adding China (see "Adding China (PROMISE) for small-project coverage")
# genuinely closed most of that gap — Nominal-complexity coverage now
# extends down to a real 1.2 KLOC / 1.4 person-month project — but this
# mechanism is kept regardless, since SOME complexity levels (e.g. "Very
# Low," which no real project in any of the three datasets has ever been
# rated) still have zero or thin real coverage. The model isn't
# malfunctioning when this fires — it's extrapolating into a region with
# no real support, and a tree model's leaf-average behavior there is not
# the same thing as a wrong prediction on a request the training data
# actually covers.
# ---------------------------------------------------------------------------
_raw_unified_df = build_unified_dataframe()  # pre-scaling — real KLOC/effort units
SOFTWARE_COMPLEXITY_COVERAGE: dict[str, dict] = {}
for _level in COCOMO_ORDINAL_SCALE:
    _subset = _raw_unified_df[_raw_unified_df[COMPLEXITY_COL] == _level]
    if len(_subset) == 0:
        SOFTWARE_COMPLEXITY_COVERAGE[_level] = {"count": 0}
        continue
    _smallest = _subset.loc[_subset[PROJECT_SIZE_COL].idxmin()]
    SOFTWARE_COMPLEXITY_COVERAGE[_level] = {
        "count": int(len(_subset)),
        "min_kloc": float(_smallest[PROJECT_SIZE_COL]),
        "min_kloc_effort": float(_smallest[SOFTWARE_TARGET_COL]),
    }
del _level, _subset  # loop vars, not meant to leak as module "constants"


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
    row's scaled value|, not raw global importance. Salience/ranking/pct
    are computed from the ABSOLUTE scaled value exactly as before — that
    part is unchanged. The SIGNED scaled value is also kept per feature
    (as "z") purely so build_factor_sentences() can state DIRECTION for
    DIRECTIONAL_FEATURES; it plays no part in the salience math or the
    ranking/ordering above, both of which stay magnitude-only.
    """
    importances = model.feature_importances_
    signed_values = transformed_row.iloc[0].to_numpy()
    salience = importances * transformed_row.iloc[0].abs().to_numpy()

    fi_df = pd.DataFrame({"feature": transformed_row.columns, "salience": salience, "z": signed_values})
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
                "z": float(row["z"]),
                "opacity": opacities[i] if i < len(opacities) else opacities[-1],
            }
        )
    return rows


def build_factor_sentences(feature_rows: list[dict], raw_row: dict) -> list[str]:
    """Numbered, plain-language list below the bar chart (Part C) — real
    per-request feature + percentage. `raw_row` is the SAME raw
    (pre-scaled) dict predict_software()/predict_construction() already
    built for the model — passed in here rather than re-derived, so this
    function never duplicates any scaling/encoding logic.

    DIRECTIONAL_FEATURES get a sentence stating whether THIS request's
    value is above/below what's typical in the training data (using the
    signed `z` rank_features_for_request now keeps, not the abs() used
    for salience) — fixes a real bug where a LOW complexity pick still
    read "higher complexity ratings... increases the estimate," which
    directly contradicted the Low pick. Every other feature keeps its
    existing per-feature-TYPE explanatory clause (see assumption 7);
    team_experience additionally gets the picked level's human-readable
    label spliced in where cheap to do so (EXPERIENCE_LABELS), without
    changing its neutral, non-directional wording.
    """
    sentences = []
    for row in feature_rows:
        raw_feature = row["raw_feature"]

        if raw_feature in DIRECTIONAL_FEATURES and raw_feature in raw_row:
            value_phrase = _format_directional_value(raw_feature, raw_row[raw_feature])
            sentences.append(_direction_sentence(row["label"], row["pct"], value_phrase, row["z"]))
            continue

        explanation = explain_feature(raw_feature)
        if raw_feature == TEAM_EXPERIENCE_COL and TEAM_EXPERIENCE_COL in raw_row:
            level_key = str(int(raw_row[TEAM_EXPERIENCE_COL]))
            level_label = EXPERIENCE_LABELS.get(level_key, "").split(" (")[0]  # drop "(little to no experience)"
            if level_label:
                explanation = f"this project's {level_label}-level {explanation}"
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

def predict_software(complexity: str, team_experience: str, kloc: float, cost_region: str) -> dict:
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

    # Honest out-of-training-range flag — see SOFTWARE_COMPLEXITY_COVERAGE
    # above. Computed from the SAME real, raw KLOC unit the form collects
    # (not the scaled/transformed vector), so the comparison is apples to
    # apples with the training data's own real project sizes.
    coverage = SOFTWARE_COMPLEXITY_COVERAGE.get(complexity, {"count": 0})
    confidence_note = None
    if coverage["count"] == 0:
        confidence_note = (
            f"No real historical project in this model's training data has "
            f"\u201c{ORDINAL_LABELS.get(complexity, complexity)}\u201d complexity at all \u2014 "
            f"this estimate isn't grounded in any comparable example, so treat it as a rough "
            f"extrapolation only."
        )
    elif kloc < coverage["min_kloc"]:
        confidence_note = (
            f"This project size ({kloc:g} KLOC) is smaller than any "
            f"\u201c{ORDINAL_LABELS.get(complexity, complexity)}\u201d-complexity project in the "
            f"real historical training data \u2014 the smallest one on record is "
            f"{coverage['min_kloc']:g} KLOC, which took {coverage['min_kloc_effort']:.0f} "
            f"person-months. This estimate extrapolates beyond what the model has actually seen, "
            f"so treat it with reduced confidence."
        )

    # Region-selectable $/person-month rate — a business-rate lookup, not
    # a model behavior (see COST_REGIONS above / results/cost_regions_
    # notes.md). Effort prediction above is entirely unaffected by region;
    # only this multiplier changes.
    cost_rate = COST_REGION_RATES[cost_region]

    return {
        "predicted_effort_months": predicted_effort_months,
        "predicted_cost": predicted_effort_months * cost_rate,
        "cost_rate": cost_rate,
        "cost_region_label": COST_REGION_LABELS[cost_region],
        "feature_rows": feature_rows,
        "raw_row": row,
        "is_construction": False,
        "confidence_note": confidence_note,
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
        "raw_row": row,
        "is_construction": True,
        "confidence_note": None,
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
        if form.get("cost_region") not in COST_REGION_VALUES:
            errors["cost_region"] = "Please choose a cost basis region."

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
        cost_region_options=COST_REGION_OPTIONS,
        month_options=MONTH_OPTIONS,
        current_year=CURRENT_YEAR,
        min_start_year=MIN_START_YEAR,
        max_start_year=MAX_START_YEAR,
        locality_options=LOCALITY_OPTIONS,
        max_project_size_kloc=MAX_PROJECT_SIZE_KLOC,
        kloc_presets=KLOC_PRESETS,
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
            cost_region=request.form["cost_region"],
        )
    else:
        result = predict_construction(request.form)

    factor_sentences = build_factor_sentences(result["feature_rows"], result["raw_row"])

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
        cost_rate=result.get("cost_rate"),
        cost_region_label=result.get("cost_region_label"),
        feature_rows=result["feature_rows"],
        factor_sentences=factor_sentences,
        confidence_note=result.get("confidence_note"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
