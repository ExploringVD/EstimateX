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
   software.pkl (the unified COCOMO-NASA + Desharnais + China model) —
   there is only one trained software model, not three. Splitting the
   dropdown into three labels is purely so a user describing their
   project picks something that feels natural, not because the
   underlying prediction differs by sub-domain; this is stated here so
   it's never mistaken for three separate models.
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
   confirming there was truly nothing to add) confirmed this.
   `source_dataset` has no real-world form equivalent either — it's a
   training-data-provenance flag, not a property of a new hypothetical
   project, so it isn't a form field at all. It's set to the MODE of the
   unified training data (computed at startup, see
   SOFTWARE_SOURCE_DEFAULT) — its feature importance is the lowest of
   the 4 (~0.005), so this
   default has minimal effect on predictions either way.

3. CONSTRUCTION FORM FIELDS. All 8 of the construction model's real
   features (locality, floor area, lot area, the three preliminary-cost
   variants, planned duration, starting unit price) are exposed as
   form fields directly; none need to be defaulted, since none are "missing" from
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
   default "United States") rather than a single constant — this is a
   safe business-rate lookup change rather than a machine-learning change
   (the effort prediction itself never changes with region, only the
   multiplier applied to it afterward). Construction's cost, by contrast,
   is the model's actual predicted target (V-10, "10000 IRR" units) —
   genuinely NOT a dollar figure, so it is shown in its own native unit
   with an explicit label, not converted or given a "$", and does NOT get
   a region selector either ("why Construction doesn't get this" — mixing
   an Iranian-Rial-denominated, already-locality-aware ML prediction with
   a generic software-salary figure would have no honest basis; see #8
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

10. REGION-SELECTABLE COST RATE — SOFTWARE DOMAINS ONLY.
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

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import (  # noqa: E402
    COCOMO_ORDINAL_COLS,
    COCOMO_ORDINAL_SCALE,
    encode_onehot,
    load_artifact,
    scale_numeric_features,
)
from preprocess_construction import (  # noqa: E402
    NUMERIC_COLS as CONSTRUCTION_NUMERIC_COLS,
    ONEHOT_COLS as CONSTRUCTION_ONEHOT_COLS,
)
from train_software_cocomo import CplxOffsetRidge  # noqa: E402,F401 — needed for joblib to unpickle SOFTWARE_MODEL

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
# spot-checked 2026-09-12 — every source and the explicit caveat is below.
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
# Values are the 25th / 50th / 75th percentile of equivphyskloc across
# COCOMO-NASA's own 93 rows (data/raw/cocomo_nasa.csv) — recomputed here
# because the deployed model is now trained on COCOMO-NASA alone, not the
# old China+Desharnais+COCOMO unified blend these presets used to be
# derived from.
KLOC_PRESETS = [
    ("small", "Small", 15.4),
    ("medium", "Medium", 47.5),
    ("large", "Large", 111),
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

# Team Experience (the single form dropdown above) maps onto FIVE of
# COCOMO-NASA's 15 drivers at once — the ones that are genuinely about
# the team's own skill/experience, not the project itself: acap (Analyst
# Capability), aexp (Applications Experience), pcap (Programmer
# Capability), vexp (Virtual Machine Experience), lexp (Programming
# Language Experience). An Advanced override for any one of these (see
# ADVANCED_DRIVER_COLS below) replaces this mapped default for that
# single driver only — the other four still follow Team Experience.
TEAM_EXPERIENCE_TO_ORDINAL = {"0": "vl", "1": "l", "2": "n", "3": "h", "4": "vh"}
EXPERIENCE_LINKED_DRIVERS = ["acap", "aexp", "pcap", "vexp", "lexp"]

# The remaining 9 of COCOMO-NASA's 15 drivers (cplx is its own top-level
# "Project Complexity" field, not here) — every one of these is exposed
# ONLY in the collapsible "Advanced cost drivers" section, defaulting to
# Nominal, per the brief.
ADVANCED_ONLY_DRIVERS = ["rely", "data", "time", "stor", "virt", "turn", "modp", "tool", "sced"]
ADVANCED_DRIVER_COLS = EXPERIENCE_LINKED_DRIVERS + ADVANCED_ONLY_DRIVERS

# Sentinel submitted for an Advanced experience-linked driver left at its
# default "Auto" option — means "use the Team Experience mapping," not a
# literal ordinal level. Only meaningful for EXPERIENCE_LINKED_DRIVERS;
# ADVANCED_ONLY_DRIVERS have no such auto-default, they're literally
# Nominal unless changed.
ADVANCED_AUTO_VALUE = ""
ADVANCED_EXPERIENCE_DRIVER_OPTIONS = [(ADVANCED_AUTO_VALUE, "Auto (from Team Experience)")] + ORDINAL_OPTIONS

# Human-readable name + one-line hint per COCOMO-81 development mode —
# shown as the "Development mode" dropdown, software domains only.
DEV_MODE_OPTIONS = [
    ("organic", "Organic", "Small, experienced team; familiar, in-house work; flexible requirements."),
    ("semidetached", "Semi-detached", "A mix of team experience and requirement flexibility — the middle ground."),
    ("embedded", "Embedded", "Tight constraints (hardware, regulations, interfaces); large or inflexible requirements."),
]
DEV_MODE_VALUES = {v for v, _, _ in DEV_MODE_OPTIONS}
DEV_MODE_LABELS = {v: label for v, label, _ in DEV_MODE_OPTIONS}
DEFAULT_DEV_MODE = "organic"

# Project category — derived from the predicted effort alone (see
# predict_software / compute_project_category below), NOT a separate
# classifier. 150 person-months (~12.5 person-years) is the reference
# point; the 120-190 PM band around it is deliberately wide (not a sharp
# cutoff at 150) because the regression itself has real error, so a
# prediction landing close to 150 on either side is reported as
# genuinely borderline rather than confidently placed in one bucket.
PROJECT_CATEGORY_LOW_MAX_PM = 120
PROJECT_CATEGORY_HIGH_MIN_PM = 190
PROJECT_CATEGORY_REFERENCE_PM = 150
PROJECT_CATEGORY_LABELS = {
    "below": "Up to 150 person-months",
    "borderline": "Borderline (around 150 person-months)",
    "above": "Above 150 person-months",
}

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
    "dev_mode": DEFAULT_DEV_MODE,
    # Advanced cost drivers — experience-linked ones default to "Auto"
    # (follow Team Experience); the rest default to Nominal ("n").
    **{f"adv_{driver}": ADVANCED_AUTO_VALUE for driver in EXPERIENCE_LINKED_DRIVERS},
    **{f"adv_{driver}": "n" for driver in ADVANCED_ONLY_DRIVERS},
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

# Friendlier labels for the feature-importance chart / factor list —
# CONSTRUCTION ONLY now (software has its own COCOMO_DRIVER_LABELS below).
FEATURE_LABELS = {
    "floor_area_m2": "Total Floor Area",
    "lot_area_m2": "Lot Area",
    "prelim_est_cost_total": "Preliminary Cost Estimate (Total)",
    "prelim_est_cost": "Preliminary Cost Estimate",
    "prelim_est_cost_equiv": "Preliminary Cost Estimate (Base-Year)",
    "planned_duration": "Planned Duration",
    "unit_price_start": "Starting Unit Price",
    "locality": "Locality Zone",
}

# Why each factor TYPE generally moves an estimate — CONSTRUCTION ONLY now.
FEATURE_EXPLANATIONS = {
    "floor_area_m2": "bigger buildings require more materials and labor, which scales the estimate up",
    "lot_area_m2": "larger lots are often associated with bigger or more complex site work",
    "prelim_est_cost_total": "the total preliminary estimate made before construction — historically a strong predictor of the final cost",
    "prelim_est_cost": "the preliminary cost estimate made before construction started — historically the single strongest predictor of the final actual cost",
    "prelim_est_cost_equiv": "the base-year-adjusted preliminary estimate, correcting for how prices shifted since the project started",
    "planned_duration": "longer planned timelines are associated with larger, more expensive projects",
    "unit_price_start": "the starting price per square meter reflects local material and labor costs at the time",
    "locality": "captures location-specific cost patterns from historical projects in this same zone",
}

# COCOMO-81's own names for its 15 effort-driver ratings, plus the two
# non-driver software inputs (kloc, mode) — SOFTWARE ONLY. Used by
# rank_cocomo_drivers/build_cocomo_factor_sentences for the "what drove
# this estimate" bars and sentences.
COCOMO_DRIVER_LABELS = {
    "rely": "Required Reliability",
    "data": "Database Size",
    "cplx": "Product Complexity",
    "time": "Execution Time Constraint",
    "stor": "Main Storage Constraint",
    "virt": "Platform Volatility",
    "turn": "Computer Turnaround Time",
    "acap": "Analyst Capability",
    "aexp": "Applications Experience",
    "pcap": "Programmer Capability",
    "vexp": "Virtual Machine Experience",
    "lexp": "Language Experience",
    "modp": "Modern Programming Practices",
    "tool": "Use of Software Tools",
    "sced": "Required Development Schedule",
    "kloc": "Project Size (KLOC)",
    "mode": "Development Mode",
}

# Why each driver generally moves the estimate — SOFTWARE ONLY, used as a
# fallback only when the per-request direction is near-zero (see
# build_cocomo_factor_sentences; the direction-aware phrasing is the
# primary wording, this is just color for the "close to typical" case).
COCOMO_DRIVER_EXPLANATIONS = {
    "rely": "how much reliability failures would cost shapes how much verification effort the project needs",
    "data": "a larger database relative to program size raises data-management effort",
    "cplx": "more intricate logic and integration work increases the estimate",
    "time": "a tighter runtime-performance constraint requires more tuning and optimization effort",
    "stor": "a tighter memory/storage constraint requires more optimization effort",
    "virt": "a more volatile underlying platform forces more rework as it changes under the project",
    "turn": "slower computer turnaround time historically cost developers more waiting/debugging time",
    "acap": "stronger analysts historically deliver the same scope in less effort",
    "aexp": "more experience with this application domain reduces ramp-up and rework",
    "pcap": "stronger programmers historically deliver the same scope in less effort",
    "vexp": "more experience with the target platform reduces ramp-up time",
    "lexp": "more experience with the implementation language reduces ramp-up time",
    "modp": "modern, disciplined programming practices historically reduce rework",
    "tool": "better software tooling historically reduces manual effort",
    "sced": "a compressed schedule relative to the project's natural pace costs extra coordination effort",
    "kloc": "larger codebases require proportionally more implementation and testing effort",
    "mode": "development mode captures how tightly constrained and how experienced/flexible the project is overall",
}

# Features whose relationship to the estimate has a clear, already-
# documented sign in FEATURE_EXPLANATIONS above ("higher X increases/
# scales up the estimate"). For these, build_factor_sentences() below
# states whether THIS request's value is above or below what's typical in
# the training data, instead of always saying "higher X increases the
# estimate" regardless of whether the actual value is low or high.
# CONSTRUCTION ONLY now — software's linear model states direction from
# the signed coefficient contribution directly (see
# build_cocomo_factor_sentences), not from this z-score mechanism.
#
# locality is deliberately NOT here: it's a categorical/provenance flag
# with no meaningful "typical value" to be above or below. Its existing
# neutral phrasing is correct and is left untouched.
DIRECTIONAL_FEATURES = {
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

SOFTWARE_MODEL = load_artifact(f"{MODELS_DIR}/final_model_software.pkl")  # CplxOffsetRidge, log(effort), COCOMO-NASA's 15 drivers
SOFTWARE_METADATA = load_artifact(f"{MODELS_DIR}/final_model_software_metadata.pkl")

CONSTRUCTION_MODEL = load_artifact(f"{MODELS_DIR}/final_model_construction.pkl")
CONSTRUCTION_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_construction.pkl")
CONSTRUCTION_ENCODER = load_artifact(f"{MODELS_DIR}/final_encoder_construction.pkl")  # OneHotEncoder, locality

# SOFTWARE_MODEL.ridge_ is the fitted Ridge(alpha=0.1, positive=True) on
# the 14 non-cplx drivers + log_kloc + mode; cplx itself is a FIXED
# weight=1 offset, never fitted (see src/train_software_cocomo.py's
# CplxOffsetRidge — this guarantees complexity's effect is Boehm's own
# monotonically-increasing published multiplier table, not whatever sign
# a free regression coefficient happens to land on under multicollinearity).
SOFTWARE_COEF = dict(zip(SOFTWARE_MODEL.feature_names_, SOFTWARE_MODEL.ridge_.coef_))
SOFTWARE_TRAINING_MEAN = SOFTWARE_METADATA["training_mean"]
SOFTWARE_EM_TABLE = SOFTWARE_METADATA["em_table"]
SOFTWARE_CPLX_OFFSET_COL = SOFTWARE_METADATA["cplx_offset_col"]
SOFTWARE_CPLX_WEIGHT = SOFTWARE_METADATA["cplx_weight"]


def get_cocomo_em(driver: str, level: str) -> float:
    """Boehm-1981 published effort multiplier for `driver` at `level`,
    falling back to the nearest level this specific driver has a
    published value for (e.g. TIME has no VL/L entries) — same lookup
    src/train_software_cocomo.py uses at training time.
    """
    defined = SOFTWARE_EM_TABLE[driver]
    if level in defined:
        return defined[level]
    idx = COCOMO_ORDINAL_SCALE.index(level)
    nearest = min(defined, key=lambda lv: abs(COCOMO_ORDINAL_SCALE.index(lv) - idx))
    return defined[nearest]

# ---------------------------------------------------------------------------
# Training-data coverage, COCOMO-NASA's own 93 rows (RAW units) — lets
# predict_software() honestly flag a request that extrapolates beyond any
# real historical example the model was actually trained on: a KLOC outside
# the real [min, max] range, or any of the 15 driver ratings at a level no
# real NASA project in the training data was ever rated at for that
# specific driver (e.g. acap/pcap have no "Very Low" or "Low" project on
# record at all — see src/train_software_cocomo.py). The model isn't
# malfunctioning when this fires — it's extrapolating into a region with
# no real support.
# ---------------------------------------------------------------------------
_cocomo_raw_df = pd.read_csv(f"{PROJECT_ROOT}/data/raw/cocomo_nasa.csv")
SOFTWARE_KLOC_COVERAGE = {
    "min_kloc": float(_cocomo_raw_df["equivphyskloc"].min()),
    "max_kloc": float(_cocomo_raw_df["equivphyskloc"].max()),
}
SOFTWARE_DRIVER_COVERAGE: dict[str, dict[str, dict]] = {}
for _driver in COCOMO_ORDINAL_COLS:
    _driver_levels: dict[str, dict] = {}
    for _level, _subset in _cocomo_raw_df.groupby(_driver):
        _example = _subset.loc[_subset["equivphyskloc"].sub(_subset["equivphyskloc"].median()).abs().idxmin()]
        _driver_levels[_level] = {
            "count": int(len(_subset)),
            "example_kloc": float(_example["equivphyskloc"]),
            "example_effort": float(_example["act_effort"]),
        }
    SOFTWARE_DRIVER_COVERAGE[_driver] = _driver_levels
del _driver, _driver_levels, _level, _subset, _example  # loop vars, not meant to leak as module "constants"


def _nearest_observed_level(driver: str, requested_level: str) -> str:
    """The closest (by ordinal distance on the vl..xh scale) level that
    DOES have real training coverage for this driver — used to name a
    real comparable example in the confidence note instead of just
    saying "no data."
    """
    observed = SOFTWARE_DRIVER_COVERAGE[driver]
    requested_idx = COCOMO_ORDINAL_SCALE.index(requested_level)
    return min(observed, key=lambda lvl: abs(COCOMO_ORDINAL_SCALE.index(lvl) - requested_idx))


# ---------------------------------------------------------------------------
# Transform-only preprocessing (reuses src/preprocessing.py's exact
# functions with the loaded, already-fitted scaler/encoders — construction
# only; software's COCOMO feature row is built by build_cocomo_feature_row
# below, driven by SOFTWARE_METADATA rather than a fitted scaler/encoder).
# ---------------------------------------------------------------------------

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

    CONSTRUCTION ONLY now — software's linear model has its own
    coefficient-based explainability (rank_cocomo_drivers), since a tree
    model's .feature_importances_ doesn't apply to it.
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


# ---------------------------------------------------------------------------
# Software (COCOMO-NASA) explainability — a log-linear model's
# coefficients give an EXACT signed per-request contribution, so this
# doesn't need the magnitude-only salience/z-score mechanism above (that
# stays construction-only, see rank_features_for_request).
# ---------------------------------------------------------------------------

COCOMO_NEAR_TYPICAL_MULTIPLIER_PCT = 3.0  # |effort multiplier| below this reads as "close to typical"


def build_cocomo_feature_row(driver_levels: dict[str, str], kloc: float, mode: str) -> pd.DataFrame:
    """The exact row src/train_software_cocomo.py's CplxOffsetRidge
    expects: log(published EM) for the 14 non-cplx drivers, log(kloc),
    one-hot `mode`, and the cplx offset column (SOFTWARE_CPLX_OFFSET_COL)
    — NOT a raw 0-5 ordinal index (see SOFTWARE_MODEL's loading comment
    above for why).
    """
    row = {
        driver: np.log(get_cocomo_em(driver, driver_levels[driver]))
        for driver in SOFTWARE_METADATA["non_cplx_drivers"]
    }
    row["log_kloc"] = np.log(kloc)
    for cat in SOFTWARE_METADATA["mode_categories"]:
        row[f"mode_{cat}"] = 1 if cat == mode else 0
    row[SOFTWARE_CPLX_OFFSET_COL] = np.log(get_cocomo_em("cplx", driver_levels["cplx"]))
    return pd.DataFrame([row])[SOFTWARE_METADATA["feature_order"] + [SOFTWARE_CPLX_OFFSET_COL]]


def rank_cocomo_drivers(driver_levels: dict[str, str], kloc: float, mode: str) -> list[dict]:
    """Per-request contribution of each LOGICAL driver (the 15 ordinal
    ratings individually, plus one combined "kloc" and one combined
    "mode" entry for their underlying raw columns — log_kloc is one
    column already, but mode_embedded/mode_organic/mode_semidetached are
    three one-hot columns that represent a single user choice, so their
    contributions are summed into one entry rather than shown as three
    separate, confusing bars) = weight x (value - training mean), on the
    log scale (see module docstring Part B2). 14 of the 15 drivers use
    their FITTED Ridge weight; `cplx` uses its FIXED weight of
    SOFTWARE_CPLX_WEIGHT (1.0) on its own published EM value, exactly
    like src/train_software_cocomo.py's CplxOffsetRidge does internally
    at prediction time — so cplx's contribution is always positive for
    above-Nominal complexity, by construction. Converted to an effort
    multiplier vs. a typical project: exp(contribution) - 1.

    Top TOP_N_FEATURES by |contribution| are returned, bar-width `pct`
    computed the same way as rank_features_for_request (share of the
    shown rows' total |contribution|, NOT the multiplier itself — the
    multiplier is reported separately in build_cocomo_factor_sentences).
    """
    mean = SOFTWARE_TRAINING_MEAN
    entries = []
    for driver in SOFTWARE_METADATA["non_cplx_drivers"]:
        level = driver_levels[driver]
        value = np.log(get_cocomo_em(driver, level))
        contribution = SOFTWARE_COEF[driver] * (value - mean[driver])
        entries.append({
            "raw_feature": driver,
            "contribution": contribution,
            "value_phrase": f"{ORDINAL_LABELS[level]} {COCOMO_DRIVER_LABELS[driver]}",
        })

    cplx_level = driver_levels["cplx"]
    cplx_value = np.log(get_cocomo_em("cplx", cplx_level))
    cplx_contribution = SOFTWARE_CPLX_WEIGHT * (cplx_value - mean[SOFTWARE_CPLX_OFFSET_COL])
    entries.append({
        "raw_feature": "cplx",
        "contribution": cplx_contribution,
        "value_phrase": f"{ORDINAL_LABELS[cplx_level]} {COCOMO_DRIVER_LABELS['cplx']}",
    })

    log_kloc = np.log(kloc)
    kloc_contribution = SOFTWARE_COEF["log_kloc"] * (log_kloc - mean["log_kloc"])
    entries.append({
        "raw_feature": "kloc",
        "contribution": kloc_contribution,
        "value_phrase": f"{kloc:g} KLOC project size",
    })

    mode_contribution = sum(
        SOFTWARE_COEF[f"mode_{cat}"] * ((1 if cat == mode else 0) - mean[f"mode_{cat}"])
        for cat in SOFTWARE_METADATA["mode_categories"]
    )
    entries.append({
        "raw_feature": "mode",
        "contribution": mode_contribution,
        "value_phrase": f"{DEV_MODE_LABELS[mode]} development mode",
    })

    entries.sort(key=lambda e: abs(e["contribution"]), reverse=True)
    top = entries[:TOP_N_FEATURES]
    total_abs = sum(abs(e["contribution"]) for e in top)

    opacities = [1, 0.78, 0.6, 0.42, 0.28]
    rows = []
    for i, entry in enumerate(top):
        pct = float(abs(entry["contribution"]) / total_abs * 100) if total_abs > 0 else 0.0
        rows.append({
            "raw_feature": entry["raw_feature"],
            "label": COCOMO_DRIVER_LABELS[entry["raw_feature"]],
            "pct": round(pct, 1),
            "contribution": entry["contribution"],
            "multiplier_pct": (float(np.exp(entry["contribution"])) - 1) * 100,
            "value_phrase": entry["value_phrase"],
            "opacity": opacities[i] if i < len(opacities) else opacities[-1],
        })
    return rows


def build_cocomo_factor_sentences(feature_rows: list[dict]) -> list[str]:
    """Direction-aware sentences from the SIGNED contribution directly
    (no z-score banding needed — a linear model's coefficient sign IS
    the direction), plus the effort multiplier the brief asks for, e.g.
    "Very High complexity: +35% effort."
    """
    sentences = []
    for row in feature_rows:
        multiplier_pct = row["multiplier_pct"]
        label, pct, value_phrase = row["label"], row["pct"], row["value_phrase"]
        if abs(multiplier_pct) < COCOMO_NEAR_TYPICAL_MULTIPLIER_PCT:
            sentences.append(
                f"{label} ({pct}%) — this project's {value_phrase} is close to typical, "
                f"so it isn't pulling the estimate strongly in either direction."
            )
            continue
        verb = "pushing" if multiplier_pct > 0 else "pulling"
        direction = "up" if multiplier_pct > 0 else "down"
        sentences.append(
            f"{label} ({pct}%) — this project's {value_phrase} is {verb} the estimate {direction} "
            f"by about {abs(multiplier_pct):.0f}% vs. a typical project."
        )
    return sentences


def build_cocomo_confidence_note(driver_levels: dict[str, str], kloc: float) -> str | None:
    """Rebuilt for the COCOMO-NASA-native model (Part B3): flags a KLOC
    outside the real training range, or any driver rated at a level no
    real NASA project was ever rated at for that specific driver —
    naming the nearest real comparable example in training data. Only
    the single most relevant issue is surfaced (KLOC first, then the
    first out-of-coverage driver in COCOMO_ORDINAL_COLS order), same
    one-note convention as the rest of this app rather than an
    overwhelming multi-paragraph list.
    """
    if kloc < SOFTWARE_KLOC_COVERAGE["min_kloc"] or kloc > SOFTWARE_KLOC_COVERAGE["max_kloc"]:
        return (
            f"This project size ({kloc:g} KLOC) is outside the real training data's range "
            f"({SOFTWARE_KLOC_COVERAGE['min_kloc']:g}–{SOFTWARE_KLOC_COVERAGE['max_kloc']:g} KLOC) — "
            f"this estimate extrapolates beyond what the model has actually seen, so treat it with "
            f"reduced confidence."
        )

    for driver in COCOMO_ORDINAL_COLS:
        level = driver_levels[driver]
        if level in SOFTWARE_DRIVER_COVERAGE[driver]:
            continue
        nearest = _nearest_observed_level(driver, level)
        example = SOFTWARE_DRIVER_COVERAGE[driver][nearest]
        return (
            f"No real historical project in this model's training data was ever rated "
            f"“{ORDINAL_LABELS[level]}” on {COCOMO_DRIVER_LABELS[driver]} — the nearest real "
            f"comparable example was rated “{ORDINAL_LABELS[nearest]}” on this driver (a "
            f"{example['example_kloc']:g} KLOC project that took {example['example_effort']:.0f} "
            f"person-months). This estimate extrapolates beyond what the model has actually seen on this "
            f"driver, so treat it with reduced confidence."
        )
    return None


def compute_project_category(predicted_effort_months: float) -> str:
    """Derived purely from the predicted effort — NOT a separate model
    (Part B4). The 120-190 PM borderline band straddles the 150 PM
    reference point deliberately wide, not a sharp cutoff, since the
    regression's own error means a prediction near 150 could honestly
    fall on either side of it.
    """
    if predicted_effort_months < PROJECT_CATEGORY_LOW_MAX_PM:
        return "below"
    if predicted_effort_months < PROJECT_CATEGORY_HIGH_MIN_PM:
        return "borderline"
    return "above"


def build_factor_sentences(feature_rows: list[dict], raw_row: dict) -> list[str]:
    """Numbered, plain-language list below the bar chart (Part C) — real
    per-request feature + percentage. CONSTRUCTION ONLY now (software has
    its own build_cocomo_factor_sentences, since its linear model's
    coefficients give an exact signed contribution rather than a z-score
    band). `raw_row` is the SAME raw (pre-scaled) dict predict_
    construction() already built for the model — passed in here rather
    than re-derived, so this function never duplicates any scaling logic.

    DIRECTIONAL_FEATURES get a sentence stating whether THIS request's
    value is above/below what's typical in the training data (using the
    signed `z` rank_features_for_request keeps, not the abs() used for
    salience). Every other feature keeps its existing per-feature-TYPE
    explanatory clause (see assumption 7).
    """
    sentences = []
    for row in feature_rows:
        raw_feature = row["raw_feature"]

        if raw_feature in DIRECTIONAL_FEATURES and raw_feature in raw_row:
            value_phrase = _format_directional_value(raw_feature, raw_row[raw_feature])
            sentences.append(_direction_sentence(row["label"], row["pct"], value_phrase, row["z"]))
            continue

        explanation = explain_feature(raw_feature)
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

def resolve_cocomo_driver_levels(form) -> dict[str, str]:
    """Builds the level (vl..xh) for all 15 COCOMO-NASA drivers from the
    form: cplx comes straight from "Project Complexity"; the 5
    EXPERIENCE_LINKED_DRIVERS follow Team Experience (TEAM_EXPERIENCE_TO_
    ORDINAL) unless an Advanced override was submitted for that specific
    driver; the remaining 9 ADVANCED_ONLY_DRIVERS take their Advanced
    value, defaulting to Nominal if somehow missing (see Part B1).
    """
    levels = {"cplx": form["complexity"]}

    mapped_default = TEAM_EXPERIENCE_TO_ORDINAL[form["team_experience"]]
    for driver in EXPERIENCE_LINKED_DRIVERS:
        override = (form.get(f"adv_{driver}") or "").strip()
        levels[driver] = override if override else mapped_default

    for driver in ADVANCED_ONLY_DRIVERS:
        levels[driver] = (form.get(f"adv_{driver}") or "").strip() or "n"

    return levels


def predict_software(form) -> dict:
    kloc = float(form["project_size_kloc"])
    mode = form.get("dev_mode", DEFAULT_DEV_MODE)
    driver_levels = resolve_cocomo_driver_levels(form)

    transformed = build_cocomo_feature_row(driver_levels, kloc, mode)
    predicted_log_effort = float(SOFTWARE_MODEL.predict(transformed)[0])
    predicted_effort_months = max(float(np.exp(predicted_log_effort)), 0.0)

    feature_rows = rank_cocomo_drivers(driver_levels, kloc, mode)
    factor_sentences = build_cocomo_factor_sentences(feature_rows)
    confidence_note = build_cocomo_confidence_note(driver_levels, kloc)
    project_category = PROJECT_CATEGORY_LABELS[compute_project_category(predicted_effort_months)]

    # Region-selectable $/person-month rate — a business-rate lookup, not
    # a model behavior (see COST_REGIONS above). Effort prediction above
    # is entirely unaffected by region; only this multiplier changes.
    cost_region = form["cost_region"]
    cost_rate = COST_REGION_RATES[cost_region]

    return {
        "predicted_effort_months": predicted_effort_months,
        "predicted_cost": predicted_effort_months * cost_rate,
        "cost_rate": cost_rate,
        "cost_region_label": COST_REGION_LABELS[cost_region],
        "feature_rows": feature_rows,
        "factor_sentences": factor_sentences,
        "project_category": project_category,
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
    factor_sentences = build_factor_sentences(feature_rows, row)

    return {
        "predicted_effort_months": None,
        "predicted_cost": predicted_cost,
        "feature_rows": feature_rows,
        "factor_sentences": factor_sentences,
        "project_category": None,
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
        if form.get("dev_mode") not in DEV_MODE_VALUES:
            errors["dev_mode"] = "Please choose a development mode."

        err = _validate_positive_number(form.get("project_size_kloc"), "Project size", max_value=MAX_PROJECT_SIZE_KLOC)
        if err:
            errors["project_size_kloc"] = err

        # Advanced cost drivers — all optional; "" (Auto) is always valid
        # for EXPERIENCE_LINKED_DRIVERS, any other submitted value must be
        # a real ordinal level for ANY of the 14 Advanced-eligible drivers.
        for driver in ADVANCED_DRIVER_COLS:
            raw_value = (form.get(f"adv_{driver}") or "").strip()
            if raw_value and raw_value not in ORDINAL_VALUES:
                errors[f"adv_{driver}"] = "Please choose a valid level, or leave this at its default."

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
        dev_mode_options=DEV_MODE_OPTIONS,
        advanced_experience_driver_options=ADVANCED_EXPERIENCE_DRIVER_OPTIONS,
        advanced_only_drivers=[(driver, COCOMO_DRIVER_LABELS[driver]) for driver in ADVANCED_ONLY_DRIVERS],
        experience_linked_drivers=[(driver, COCOMO_DRIVER_LABELS[driver]) for driver in EXPERIENCE_LINKED_DRIVERS],
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
        result = predict_software(request.form)
    else:
        result = predict_construction(request.form)

    factor_sentences = result["factor_sentences"]

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
        project_category=result.get("project_category"),
        project_category_reference_pm=PROJECT_CATEGORY_REFERENCE_PM,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
