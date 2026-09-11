"""
EstimateX — Software domain unification.

Combines COCOMO-NASA and Desharnais into ONE harmonized software-effort
dataset (data/processed/software_unified.csv), so the two "software"
project types share a single trained model instead of two separate ones.
This makes "multiple domains" a real, single-model-per-domain design
(software vs. construction) rather than software secretly being two
domains wearing one label.

=====================================================================
COLUMN-MAPPING DECISIONS — every one documented here, for the paper:
=====================================================================

Unified schema: team_experience, project_size_kloc, complexity,
source_dataset, effort_person_months (target). Chosen to match exactly
what the roadmap asked for (team size/experience, project size,
complexity, target=effort) — not a superset stuffed with COCOMO-only
columns that would just be constant-imputed padding on every Desharnais
row.

1. team_experience
   - COCOMO-NASA has no single number for this — it has 5 separate
     ordinal experience/capability ratings (acap, aexp, pcap, vexp,
     lexp), each on the vl/l/n/h/vh/xh scale. Ordinal-encoded to 0-5
     (same scale/order already used throughout this project) and
     averaged into one team_experience proxy per row.
   - Desharnais already has two numeric experience measures in years
     (TeamExp, ManagerExp, both confirmed 0-4 in the raw data). Averaged
     into one number the same way.
   - The two source scales aren't identical (COCOMO 0-5 ordinal ratings
     vs. Desharnais 0-4 years) — StandardScaler is fit on the COMBINED
     unified column afterward, which normalizes both onto the same
     final distribution regardless of their differing raw ranges.

2. project_size_kloc
   - COCOMO-NASA: `equivphyskloc` used directly — already KLOC, no
     conversion needed.
   - Desharnais has no lines-of-code column at all; it measures size in
     function points. Converted to an approximate KLOC-equivalent using
     the same Capers Jones "~100 LOC per function point" backfire
     heuristic already used in app.py's live Desharnais prediction path
     (BACKFIRE_LOC_PER_FP, now shared from preprocessing.py) — kept
     identical here for methodological consistency across the whole
     project. `PointsAdjust` (the dataset's primary/adjusted function
     point count) is the column converted.

3. complexity
   - COCOMO-NASA: `cplx` used directly — already exactly this, same
     vl-xh scale, ordinal-encoded to 0-5.
   - Desharnais has NO complexity-like column at all (already
     established in app.py's docstring for the live app). There is no
     honest way to derive one from Desharnais's own columns, so it is
     imputed with COCOMO-NASA's MODAL complexity rating (the most common
     value among COCOMO-NASA's 93 projects) — a stated, documented
     default, not a fabricated measurement.

4. source_dataset
   - A binary flag (0 = COCOMO-NASA, 1 = Desharnais), added as its own
     feature per the roadmap's suggested fallback option. Lets the
     unified model learn any systematic difference between the two data
     collection contexts (including implicitly "this row's complexity
     value is imputed, not measured") rather than the imputation in (3)
     silently masquerading as real signal.

5. effort_person_months (target)
   - COCOMO-NASA's `act_effort` is already person-months — used as is.
   - Desharnais's `Effort` is person-hours — converted to person-months
     using the same Boehm/COCOMO HOURS_PER_PERSON_MONTH=152 constant
     already used in app.py, now shared from preprocessing.py.

Rows with missing values are handled exactly as Step 4 already
established per dataset (COCOMO-NASA: none; Desharnais: the 4 rows with
'?' in TeamExp/ManagerExp are dropped) — reusing
preprocessing.handle_missing_values, not reimplemented.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from preprocessing import (
    BACKFIRE_LOC_PER_FP,
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
    HOURS_PER_PERSON_MONTH,
    handle_missing_values,
    load_raw_csv,
    save_artifact,
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

OUTPUT_PATH = f"{PROCESSED_DIR}/software_unified.csv"

TEAM_EXPERIENCE_COL = "team_experience"
PROJECT_SIZE_COL = "project_size_kloc"
COMPLEXITY_COL = "complexity"
SOURCE_COL = "source_dataset"
TARGET_COL = "effort_person_months"

UNIFIED_NUMERIC_COLS = [TEAM_EXPERIENCE_COL, PROJECT_SIZE_COL]
# complexity is ordinal-encoded then scaled alongside the numeric columns,
# same pattern COCOMO's own ordinal drivers already used in Step 4.
SCALE_COLS = UNIFIED_NUMERIC_COLS + [COMPLEXITY_COL]

_COCOMO_EXPERIENCE_COLS = ["acap", "aexp", "pcap", "vexp", "lexp"]
_ORDINAL_INDEX = {label: i for i, label in enumerate(COCOMO_ORDINAL_SCALE)}


def _load_cocomo_rows() -> pd.DataFrame:
    raw_df = load_raw_csv(f"{RAW_DIR}/cocomo_nasa.csv")
    df = raw_df.drop(columns=[COCOMO_ID_COL])
    categorical_cols = COCOMO_ONEHOT_COLS + COCOMO_ORDINAL_COLS + [COCOMO_BINARY_COL]
    df, _ = handle_missing_values(df, numeric_cols=COCOMO_NUMERIC_COLS, categorical_cols=categorical_cols)

    experience_idx = df[_COCOMO_EXPERIENCE_COLS].apply(lambda col: col.map(_ORDINAL_INDEX))
    team_experience = experience_idx.mean(axis=1)

    unified = pd.DataFrame(
        {
            TEAM_EXPERIENCE_COL: team_experience,
            PROJECT_SIZE_COL: df["equivphyskloc"].astype(float),
            COMPLEXITY_COL: df["cplx"],  # ordinal-encoded later, alongside Desharnais's imputed value
            SOURCE_COL: 0,
            TARGET_COL: df[COCOMO_TARGET_COL].astype(float),
        }
    )
    return unified


def _load_desharnais_rows() -> pd.DataFrame:
    raw_df = load_raw_csv(f"{RAW_DIR}/desharnais.csv")
    df = raw_df.drop(columns=[DESHARNAIS_ID_COL])
    df, _ = handle_missing_values(df, numeric_cols=DESHARNAIS_NUMERIC_COLS, categorical_cols=DESHARNAIS_ONEHOT_COLS)

    team_experience = df[["TeamExp", "ManagerExp"]].astype(float).mean(axis=1)
    project_size_kloc = (df["PointsAdjust"].astype(float) * BACKFIRE_LOC_PER_FP) / 1000.0
    effort_months = df[DESHARNAIS_TARGET_COL].astype(float) / HOURS_PER_PERSON_MONTH

    # Imputed complexity: Desharnais has no analogous column at all. Filled
    # in by the caller with COCOMO-NASA's modal `cplx` value, so the
    # imputation source is explicit at the call site, not buried here.
    unified = pd.DataFrame(
        {
            TEAM_EXPERIENCE_COL: team_experience,
            PROJECT_SIZE_COL: project_size_kloc,
            COMPLEXITY_COL: pd.NA,
            SOURCE_COL: 1,
            TARGET_COL: effort_months,
        }
    )
    return unified


def build_unified_dataframe() -> pd.DataFrame:
    cocomo_rows = _load_cocomo_rows()
    desharnais_rows = _load_desharnais_rows()

    # Impute Desharnais's missing complexity with COCOMO-NASA's mode —
    # documented decision #3 above.
    modal_complexity = cocomo_rows[COMPLEXITY_COL].mode(dropna=True).iloc[0]
    desharnais_rows[COMPLEXITY_COL] = desharnais_rows[COMPLEXITY_COL].fillna(modal_complexity)

    combined = pd.concat([cocomo_rows, desharnais_rows], ignore_index=True)
    return combined


def preprocess_unified(df: pd.DataFrame) -> tuple[pd.DataFrame, OrdinalEncoder, StandardScaler]:
    df = df.copy()

    ordinal_encoder = OrdinalEncoder(categories=[COCOMO_ORDINAL_SCALE])
    df[[COMPLEXITY_COL]] = ordinal_encoder.fit_transform(df[[COMPLEXITY_COL]])

    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    # source_dataset is a 0/1 flag — left unscaled, same reasoning as
    # COCOMO's binary `forg` column in Step 4 (scaling a 0/1 indicator
    # has no meaning).
    target = df.pop(TARGET_COL)
    df[TARGET_COL] = target
    return df, ordinal_encoder, scaler


if __name__ == "__main__":
    print("EstimateX — software domain unification")
    print()

    raw_unified = build_unified_dataframe()
    print(f"COCOMO-NASA rows: {(raw_unified[SOURCE_COL] == 0).sum()}")
    print(f"Desharnais rows:  {(raw_unified[SOURCE_COL] == 1).sum()}")
    print(f"Combined rows:    {len(raw_unified)}")
    print()

    processed, ordinal_encoder, scaler = preprocess_unified(raw_unified)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}  shape={processed.shape}")

    save_artifact(scaler, f"{MODELS_DIR}/software_unified_scaler.joblib")
    save_artifact(ordinal_encoder, f"{MODELS_DIR}/software_unified_ordinal_encoder.joblib")
    print(f"Saved -> {MODELS_DIR}/software_unified_scaler.joblib")
    print(f"Saved -> {MODELS_DIR}/software_unified_ordinal_encoder.joblib")
