"""EstimateX Flask app: software effort/cost and construction cost estimates with explanations.

Loads the deployed models and fitted preprocessing from models/ (never refit here).
Run from the project root: python app/app.py
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

from preprocessing import encode_onehot, load_artifact, scale_numeric_features  # noqa: E402
from preprocess_construction import NUMERIC_COLS as CONSTRUCTION_NUMERIC_COLS  # noqa: E402
from preprocess_construction import ONEHOT_COLS as CONSTRUCTION_ONEHOT_COLS  # noqa: E402
# Importing train_software_cocomo also registers CplxOffsetRidge, which joblib needs to load the model.
from train_software_cocomo import COCOMO_ORDINAL_COLS, COCOMO_ORDINAL_SCALE, get_em  # noqa: E402

app = Flask(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# $/person-month (2026 salary / vendor-rate references). Software only: cost = effort x rate,
# while construction's cost is the model's own prediction.
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
COST_REGION_OPTIONS = [(v, label) for v, label, _ in COST_REGIONS]
DEFAULT_COST_REGION = "us"
INR_PER_USD = 85  # static display rate, not a live exchange rate

TOP_N_FEATURES = 5
FACTOR_OPACITIES = [1, 0.78, 0.6, 0.42, 0.28]
MAX_PROJECT_SIZE_KLOC = 100_000

# 25th/50th/75th percentile KLOC of the COCOMO-NASA training projects.
KLOC_PRESETS = [
    ("small", "Small", 15.4),
    ("medium", "Medium", 47.5),
    ("large", "Large", 111),
]

SOFTWARE_DOMAIN = "software"
DOMAIN_OPTIONS = [(SOFTWARE_DOMAIN, "Software"), ("construction", "Construction")]
DOMAIN_LABELS = dict(DOMAIN_OPTIONS)

ORDINAL_OPTIONS = [
    ("vl", "Very Low"), ("l", "Low"), ("n", "Nominal"),
    ("h", "High"), ("vh", "Very High"), ("xh", "Extra High"),
]
ORDINAL_LABELS = dict(ORDINAL_OPTIONS)
ORDINAL_VALUES = set(ORDINAL_LABELS)
EXPERIENCE_OPTIONS = [
    ("0", "Novice (little to no experience)"),
    ("1", "Junior"),
    ("2", "Intermediate"),
    ("3", "Senior"),
    ("4", "Expert"),
]
EXPERIENCE_VALUES = {v for v, _ in EXPERIENCE_OPTIONS}
LOCALITY_OPTIONS = list(range(1, 21))
LOCALITY_VALUES = {str(v) for v in LOCALITY_OPTIONS}

# Team Experience sets these five team-skill drivers at once; an Advanced override replaces
# the mapped level for that driver only. "" (Auto) means "follow Team Experience".
TEAM_EXPERIENCE_TO_ORDINAL = {"0": "vl", "1": "l", "2": "n", "3": "h", "4": "vh"}
EXPERIENCE_LINKED_DRIVERS = ["acap", "aexp", "pcap", "vexp", "lexp"]
ADVANCED_ONLY_DRIVERS = ["rely", "data", "time", "stor", "virt", "turn", "modp", "tool", "sced"]
ADVANCED_AUTO_VALUE = ""
ADVANCED_EXPERIENCE_DRIVER_OPTIONS = [(ADVANCED_AUTO_VALUE, "Auto (from Team Experience)")] + ORDINAL_OPTIONS

DEV_MODE_OPTIONS = [
    ("organic", "Organic", "Small, experienced team; familiar, in-house work; flexible requirements."),
    ("semidetached", "Semi-detached", "A mix of team experience and requirement flexibility — the middle ground."),
    ("embedded", "Embedded", "Tight constraints (hardware, regulations, interfaces); large or inflexible requirements."),
]
DEV_MODE_VALUES = {v for v, _, _ in DEV_MODE_OPTIONS}
DEV_MODE_LABELS = {v: label for v, label, _ in DEV_MODE_OPTIONS}

# Published COCOMO schedule equation (Boehm 1981), not a learned model:
# duration_months = 2.5 x effort_PM ^ c, with c set by development mode.
SCHEDULE_COEFFICIENT = 2.5
SCHEDULE_EXPONENTS = {"organic": 0.38, "semidetached": 0.35, "embedded": 0.32}

# Category is a threshold on predicted effort, not a separate classifier. The 120-190 PM band
# around 150 PM is wide on purpose: the regression's error makes predictions near 150 ambiguous.
PROJECT_CATEGORY_LOW_MAX_PM = 120
PROJECT_CATEGORY_HIGH_MIN_PM = 190
PROJECT_CATEGORY_REFERENCE_PM = 150
PROJECT_CATEGORY_LABELS = {
    "below": "Up to 150 person-months",
    "borderline": "Borderline (around 150 person-months)",
    "above": "Above 150 person-months",
}

MONTH_OPTIONS = [
    ("1", "January"), ("2", "February"), ("3", "March"), ("4", "April"),
    ("5", "May"), ("6", "June"), ("7", "July"), ("8", "August"),
    ("9", "September"), ("10", "October"), ("11", "November"), ("12", "December"),
]
MONTH_NAMES = dict(MONTH_OPTIONS)
CURRENT_YEAR = date.today().year
MIN_START_YEAR = 1990
MAX_START_YEAR = CURRENT_YEAR + 50

DEFAULT_FORM_VALUES = {
    "domain": SOFTWARE_DOMAIN,
    "team_experience": "2",
    "project_size_kloc": "50",
    "complexity": "n",
    "dev_mode": "organic",
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

CONSTRUCTION_LABELS = {
    "floor_area_m2": "Total Floor Area",
    "lot_area_m2": "Lot Area",
    "prelim_est_cost_total": "Preliminary Cost Estimate (Total)",
    "prelim_est_cost": "Preliminary Cost Estimate",
    "prelim_est_cost_equiv": "Preliminary Cost Estimate (Base-Year)",
    "planned_duration": "Planned Duration",
    "unit_price_start": "Starting Unit Price",
    "locality": "Locality Zone",
}
CONSTRUCTION_VALUE_PHRASES = {
    "floor_area_m2": "{:g} m² floor area",
    "lot_area_m2": "{:g} m² lot area",
    "prelim_est_cost_total": "preliminary cost estimate (total) of {:g} (×10,000,000 IRR)",
    "prelim_est_cost": "preliminary cost estimate of {:g} (×10,000 IRR)",
    "prelim_est_cost_equiv": "base-year preliminary cost estimate of {:g} (×10,000 IRR)",
    "planned_duration": "planned duration of {:g}",
    "unit_price_start": "starting unit price of {:g} (×10,000 IRR per m²)",
}
LOCALITY_EXPLANATION = "captures location-specific cost patterns from historical projects in this same zone"
NEAR_TYPICAL_Z = 0.2  # |scaled value| below this reads as "close to typical"
STRONG_Z = 1.0  # at or above this reads as "well above/below"

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
NEAR_TYPICAL_MULTIPLIER_PCT = 3.0  # effort effect below this reads as "close to typical"

SOFTWARE_MODEL = load_artifact(f"{MODELS_DIR}/final_model_software.pkl")
SOFTWARE_METADATA = load_artifact(f"{MODELS_DIR}/final_model_software_metadata.pkl")
CONSTRUCTION_MODEL = load_artifact(f"{MODELS_DIR}/final_model_construction.pkl")
CONSTRUCTION_SCALER = load_artifact(f"{MODELS_DIR}/final_scaler_construction.pkl")
CONSTRUCTION_ENCODER = load_artifact(f"{MODELS_DIR}/final_encoder_construction.pkl")

SOFTWARE_COEF = dict(zip(SOFTWARE_MODEL.feature_names_, SOFTWARE_MODEL.ridge_.coef_))
SOFTWARE_TRAINING_MEAN = SOFTWARE_METADATA["training_mean"]
SOFTWARE_CPLX_OFFSET_COL = SOFTWARE_METADATA["cplx_offset_col"]
SOFTWARE_CPLX_WEIGHT = SOFTWARE_METADATA["cplx_weight"]
# Likely range = predicted x exp(q25..q75) of the model's out-of-fold log errors.
SOFTWARE_RANGE_LOW_FACTOR = float(np.exp(SOFTWARE_METADATA["cv_log_residual_q25"]))
SOFTWARE_RANGE_HIGH_FACTOR = float(np.exp(SOFTWARE_METADATA["cv_log_residual_q75"]))


def _software_coverage(raw_df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Per driver and rated level: project count and the median-KLOC example project."""
    coverage = {}
    for driver in COCOMO_ORDINAL_COLS:
        coverage[driver] = {}
        for level, subset in raw_df.groupby(driver):
            example = subset.loc[subset["equivphyskloc"].sub(subset["equivphyskloc"].median()).abs().idxmin()]
            coverage[driver][level] = {
                "count": int(len(subset)),
                "example_kloc": float(example["equivphyskloc"]),
                "example_effort": float(example["act_effort"]),
            }
    return coverage


_cocomo_raw_df = pd.read_csv(f"{PROJECT_ROOT}/data/raw/cocomo_nasa.csv")
SOFTWARE_KLOC_RANGE = (float(_cocomo_raw_df["equivphyskloc"].min()), float(_cocomo_raw_df["equivphyskloc"].max()))
SOFTWARE_DRIVER_COVERAGE = _software_coverage(_cocomo_raw_df)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def construction_label(raw_name: str) -> str:
    if raw_name in CONSTRUCTION_LABELS:
        return CONSTRUCTION_LABELS[raw_name]
    prefix, _, suffix = raw_name.partition("_")  # one-hot column, e.g. locality_5
    return f"{CONSTRUCTION_LABELS[prefix]}: {suffix}"


def rank_construction_features(transformed_row: pd.DataFrame) -> list[dict]:
    """Top features by importance x |scaled value| (a heuristic, not SHAP)."""
    salience = CONSTRUCTION_MODEL.feature_importances_ * transformed_row.iloc[0].abs().to_numpy()
    fi_df = pd.DataFrame({"feature": transformed_row.columns, "salience": salience, "z": transformed_row.iloc[0].to_numpy()})
    fi_df = fi_df.sort_values("salience", ascending=False).head(TOP_N_FEATURES).reset_index(drop=True)

    total = fi_df["salience"].sum()
    return [
        {
            "raw_feature": row["feature"],
            "label": construction_label(row["feature"]),
            "pct": round(float(row["salience"] / total * 100) if total > 0 else 0.0, 1),
            "z": float(row["z"]),
            "opacity": FACTOR_OPACITIES[i],
        }
        for i, row in fi_df.iterrows()
    ]


def construction_factor_sentences(feature_rows: list[dict], raw_row: dict) -> list[str]:
    sentences = []
    for row in feature_rows:
        label, pct, z = row["label"], row["pct"], row["z"]
        if row["raw_feature"] not in CONSTRUCTION_VALUE_PHRASES:
            sentences.append(f"{label} ({pct}%) — {LOCALITY_EXPLANATION}.")
            continue
        value_phrase = CONSTRUCTION_VALUE_PHRASES[row["raw_feature"]].format(raw_row[row["raw_feature"]])
        if abs(z) < NEAR_TYPICAL_Z:
            sentences.append(
                f"{label} ({pct}%) — this project's {value_phrase} is close to what's typical "
                f"in the training data, so it isn't pulling the estimate strongly in either direction."
            )
            continue
        intensity = "well " if abs(z) >= STRONG_Z else ""
        comparison, verb, direction = ("below", "pulling", "down") if z < 0 else ("above", "pushing", "up")
        sentences.append(
            f"{label} ({pct}%) — this project's {value_phrase} is {intensity}{comparison} what's "
            f"typical in the training data, which is {verb} this estimate {direction}."
        )
    return sentences


def predict_construction(form) -> dict:
    row = {"locality": int(form["locality"])}
    row.update({col: float(form[col]) for col in CONSTRUCTION_NUMERIC_COLS})
    df, _ = encode_onehot(pd.DataFrame([row]), CONSTRUCTION_ONEHOT_COLS, encoder=CONSTRUCTION_ENCODER)
    df, _ = scale_numeric_features(df, CONSTRUCTION_NUMERIC_COLS, scaler=CONSTRUCTION_SCALER)
    transformed = df[CONSTRUCTION_MODEL.feature_names_in_]

    predicted_cost = max(float(CONSTRUCTION_MODEL.predict(transformed)[0]), 0.0)
    feature_rows = rank_construction_features(transformed)
    return {
        "is_construction": True,
        "predicted_cost_native": round(predicted_cost, 0),
        "feature_rows": feature_rows,
        "factor_sentences": construction_factor_sentences(feature_rows, row),
    }


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------

def resolve_cocomo_driver_levels(form) -> dict[str, str]:
    levels = {"cplx": form["complexity"]}
    mapped_default = TEAM_EXPERIENCE_TO_ORDINAL[form["team_experience"]]
    for driver in EXPERIENCE_LINKED_DRIVERS:
        levels[driver] = (form.get(f"adv_{driver}") or "").strip() or mapped_default
    for driver in ADVANCED_ONLY_DRIVERS:
        levels[driver] = (form.get(f"adv_{driver}") or "").strip() or "n"
    return levels


def build_cocomo_feature_row(driver_levels: dict[str, str], kloc: float, mode: str) -> pd.DataFrame:
    row = {driver: np.log(get_em(driver, driver_levels[driver])) for driver in SOFTWARE_METADATA["non_cplx_drivers"]}
    row["log_kloc"] = np.log(kloc)
    for cat in SOFTWARE_METADATA["mode_categories"]:
        row[f"mode_{cat}"] = 1 if cat == mode else 0
    row[SOFTWARE_CPLX_OFFSET_COL] = np.log(get_em("cplx", driver_levels["cplx"]))
    return pd.DataFrame([row])[SOFTWARE_METADATA["feature_order"] + [SOFTWARE_CPLX_OFFSET_COL]]


def rank_cocomo_drivers(driver_levels: dict[str, str], kloc: float, mode: str) -> list[dict]:
    """Top drivers by exact log-scale contribution, weight x (value - training mean)."""
    # cplx uses its fixed offset weight of 1; the three mode columns count as one entry.
    mean = SOFTWARE_TRAINING_MEAN
    entries = []
    for driver in SOFTWARE_METADATA["non_cplx_drivers"]:
        level = driver_levels[driver]
        contribution = SOFTWARE_COEF[driver] * (np.log(get_em(driver, level)) - mean[driver])
        entries.append((driver, contribution, f"{ORDINAL_LABELS[level]} {COCOMO_DRIVER_LABELS[driver]}"))

    cplx_level = driver_levels["cplx"]
    cplx_contribution = SOFTWARE_CPLX_WEIGHT * (np.log(get_em("cplx", cplx_level)) - mean[SOFTWARE_CPLX_OFFSET_COL])
    entries.append(("cplx", cplx_contribution, f"{ORDINAL_LABELS[cplx_level]} {COCOMO_DRIVER_LABELS['cplx']}"))

    kloc_contribution = SOFTWARE_COEF["log_kloc"] * (np.log(kloc) - mean["log_kloc"])
    entries.append(("kloc", kloc_contribution, f"{kloc:g} KLOC project size"))

    mode_contribution = sum(
        SOFTWARE_COEF[f"mode_{cat}"] * ((1 if cat == mode else 0) - mean[f"mode_{cat}"])
        for cat in SOFTWARE_METADATA["mode_categories"]
    )
    entries.append(("mode", mode_contribution, f"{DEV_MODE_LABELS[mode]} development mode"))

    entries.sort(key=lambda e: abs(e[1]), reverse=True)
    top = entries[:TOP_N_FEATURES]
    total_abs = sum(abs(c) for _, c, _ in top)
    return [
        {
            "label": COCOMO_DRIVER_LABELS[name],
            "pct": round(float(abs(contribution) / total_abs * 100) if total_abs > 0 else 0.0, 1),
            "multiplier_pct": (float(np.exp(contribution)) - 1) * 100,
            "value_phrase": value_phrase,
            "opacity": FACTOR_OPACITIES[i],
        }
        for i, (name, contribution, value_phrase) in enumerate(top)
    ]


def cocomo_factor_sentences(feature_rows: list[dict]) -> list[str]:
    sentences = []
    for row in feature_rows:
        multiplier_pct = row["multiplier_pct"]
        label, pct, value_phrase = row["label"], row["pct"], row["value_phrase"]
        if abs(multiplier_pct) < NEAR_TYPICAL_MULTIPLIER_PCT:
            sentences.append(
                f"{label} ({pct}%) — this project's {value_phrase} is close to typical, "
                f"so it isn't pulling the estimate strongly in either direction."
            )
            continue
        verb, direction = ("pushing", "up") if multiplier_pct > 0 else ("pulling", "down")
        sentences.append(
            f"{label} ({pct}%) — this project's {value_phrase} is {verb} the estimate {direction} "
            f"by about {abs(multiplier_pct):.0f}% vs. a typical project."
        )
    return sentences


def cocomo_confidence_note(driver_levels: dict[str, str], kloc: float) -> str | None:
    """Flags a KLOC outside the training range, else the first driver level no training project had."""
    min_kloc, max_kloc = SOFTWARE_KLOC_RANGE
    if kloc < min_kloc or kloc > max_kloc:
        return (
            f"This project size ({kloc:g} KLOC) is outside the real training data's range "
            f"({min_kloc:g}–{max_kloc:g} KLOC) — "
            f"this estimate extrapolates beyond what the model has actually seen, so treat it with "
            f"reduced confidence."
        )

    for driver in COCOMO_ORDINAL_COLS:
        level = driver_levels[driver]
        observed = SOFTWARE_DRIVER_COVERAGE[driver]
        if level in observed:
            continue
        idx = COCOMO_ORDINAL_SCALE.index(level)
        nearest = min(observed, key=lambda lvl: abs(COCOMO_ORDINAL_SCALE.index(lvl) - idx))
        example = observed[nearest]
        return (
            f"No real historical project in this model's training data was ever rated "
            f"“{ORDINAL_LABELS[level]}” on {COCOMO_DRIVER_LABELS[driver]} — the nearest real "
            f"comparable example was rated “{ORDINAL_LABELS[nearest]}” on this driver (a "
            f"{example['example_kloc']:g} KLOC project that took {example['example_effort']:.0f} "
            f"person-months). This estimate extrapolates beyond what the model has actually seen on this "
            f"driver, so treat it with reduced confidence."
        )
    return None


def compute_project_category(effort_pm: float) -> str:
    if effort_pm < PROJECT_CATEGORY_LOW_MAX_PM:
        return "below"
    if effort_pm < PROJECT_CATEGORY_HIGH_MIN_PM:
        return "borderline"
    return "above"


def compute_schedule(effort_pm: float, mode: str) -> tuple[float, float]:
    """(duration in months, average team size) from the COCOMO schedule equation."""
    duration_months = SCHEDULE_COEFFICIENT * effort_pm ** SCHEDULE_EXPONENTS[mode]
    return duration_months, effort_pm / duration_months


def compute_estimated_completion(start_month: int, start_year: int, duration_months: float) -> str:
    total_months = (start_month - 1) + round(duration_months)
    return f"{MONTH_NAMES[str(total_months % 12 + 1)]} {start_year + total_months // 12}"


def predict_software(form) -> dict:
    kloc = float(form["project_size_kloc"])
    mode = form["dev_mode"]
    driver_levels = resolve_cocomo_driver_levels(form)

    log_effort = float(SOFTWARE_MODEL.predict(build_cocomo_feature_row(driver_levels, kloc, mode))[0])
    effort = max(float(np.exp(log_effort)), 0.0)
    effort_low, effort_high = effort * SOFTWARE_RANGE_LOW_FACTOR, effort * SOFTWARE_RANGE_HIGH_FACTOR
    duration_months, avg_team_size = compute_schedule(effort, mode)

    cost_region = form["cost_region"]
    cost_rate = COST_REGION_RATES[cost_region]
    cost = effort * cost_rate
    cost_low, cost_high = effort_low * cost_rate, effort_high * cost_rate

    feature_rows = rank_cocomo_drivers(driver_levels, kloc, mode)
    return {
        "is_construction": False,
        "predicted_effort_months": round(effort, 1),
        "effort_range": (round(effort_low), round(effort_high)),
        "project_category": PROJECT_CATEGORY_LABELS[compute_project_category(effort)],
        "duration_months": round(duration_months, 1),
        "duration_range": (round(compute_schedule(effort_low, mode)[0], 1), round(compute_schedule(effort_high, mode)[0], 1)),
        "schedule_exponent": SCHEDULE_EXPONENTS[mode],
        "dev_mode_label": DEV_MODE_LABELS[mode],
        "estimated_completion": compute_estimated_completion(int(form["start_month"]), int(form["start_year"]), duration_months),
        "avg_team_size": round(avg_team_size, 1),
        "predicted_cost_usd": round(cost, 0),
        "predicted_cost_inr": round(cost * INR_PER_USD, 0),
        "cost_range_usd": (round(cost_low), round(cost_high)),
        "cost_range_inr": (round(cost_low * INR_PER_USD), round(cost_high * INR_PER_USD)),
        "cost_rate": cost_rate,
        "cost_region_label": COST_REGION_LABELS[cost_region],
        "feature_rows": feature_rows,
        "factor_sentences": cocomo_factor_sentences(feature_rows),
        "confidence_note": cocomo_confidence_note(driver_levels, kloc),
    }


# ---------------------------------------------------------------------------
# Validation and routes
# ---------------------------------------------------------------------------

def _validate_positive_number(raw_value, label: str, max_value: float | None = None) -> str | None:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return f"{label} is required."
    try:
        value = float(raw_value)
    except ValueError:
        return f"{label} must be a number."
    if value <= 0:
        return f"{label} must be a positive number."
    if max_value is not None and value > max_value:
        return f"{label} must be {max_value:,.0f} or fewer."
    return None


def validate_form(form) -> dict:
    """Field -> error message for the submitted domain's fields; empty means valid."""
    errors = {}
    domain = form.get("domain")
    if domain not in DOMAIN_LABELS:
        return {"domain": "Please choose a project domain."}

    if domain == SOFTWARE_DOMAIN:
        if form.get("complexity") not in ORDINAL_VALUES:
            errors["complexity"] = "Please choose a complexity level."
        if form.get("team_experience") not in EXPERIENCE_VALUES:
            errors["team_experience"] = "Please choose a team experience level."
        if form.get("cost_region") not in COST_REGION_RATES:
            errors["cost_region"] = "Please choose a cost basis region."
        if form.get("dev_mode") not in DEV_MODE_VALUES:
            errors["dev_mode"] = "Please choose a development mode."

        err = _validate_positive_number(form.get("project_size_kloc"), "Project size", max_value=MAX_PROJECT_SIZE_KLOC)
        if err:
            errors["project_size_kloc"] = err

        for driver in EXPERIENCE_LINKED_DRIVERS + ADVANCED_ONLY_DRIVERS:
            raw_value = (form.get(f"adv_{driver}") or "").strip()
            if raw_value and raw_value not in ORDINAL_VALUES:
                errors[f"adv_{driver}"] = "Please choose a valid level, or leave this at its default."

        if form.get("start_month") not in MONTH_NAMES:
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

    else:
        if form.get("locality") not in LOCALITY_VALUES:
            errors["locality"] = "Please choose a locality zone."
        for field, label in [
            ("floor_area_m2", "Floor area"),
            ("lot_area_m2", "Lot area"),
            ("prelim_est_cost_total", "Preliminary estimated cost (total)"),
            ("prelim_est_cost", "Preliminary estimated cost"),
            ("prelim_est_cost_equiv", "Preliminary estimated cost (base-year)"),
            ("planned_duration", "Planned duration"),
            ("unit_price_start", "Starting unit price"),
        ]:
            err = _validate_positive_number(form.get(field), label)
            if err:
                errors[field] = err

    return errors


@app.template_filter("inr")
def format_inr(value) -> str:
    """Indian digit grouping (12,34,56,789), matching the page's en-IN ₹ counter."""
    digits = str(int(round(value)))
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(groups + [tail])


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
    result = predict_software(request.form) if domain == SOFTWARE_DOMAIN else predict_construction(request.form)
    return render_template(
        "results.html",
        domain_label=DOMAIN_LABELS[domain],
        inr_per_usd=INR_PER_USD,
        project_category_reference_pm=PROJECT_CATEGORY_REFERENCE_PM,
        **result,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
