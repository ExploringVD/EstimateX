"""
EstimateX — Software domain unification.

Combines COCOMO-NASA, Desharnais, and China (PROMISE repository — added
later, see decision #6 below) into ONE harmonized software-effort dataset
(data/processed/software_unified.csv), so all "software" project types
share a single trained model instead of several separate ones. This makes
"multiple domains" a real, single-model-per-domain design (software vs.
construction) rather than software secretly being several domains
wearing one label.

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
     imputed from COCOMO-NASA's own complexity distribution — a stated,
     documented default, not a fabricated measurement.
   - REVISED (root-cause fix, see results/software_unification_notes.md
     "Fixing unrealistic small-project predictions"): this used to fill
     ALL 77 Desharnais rows with COCOMO-NASA's single MODAL value ('h').
     That collapsed 79% of the unified 170-row dataset onto one
     complexity value, starving the other levels — especially 'n'
     (Nominal) — of enough real, size-diverse examples for a tree model
     to learn a genuine size-vs-effort trend within them. A tiny/nominal
     query (e.g. 1 KLOC, Nominal) then fell into whatever sparse leaf
     the few real 'n' rows created, which happened to be anchored on a
     10-KLOC/48-month project — producing a wildly inflated prediction
     regardless of how small the actual input was.
     Fixed by sampling each Desharnais row's imputed complexity from
     COCOMO-NASA's own empirical complexity distribution (seeded,
     reproducible — RNG_SEED below) instead of a single constant. This
     preserves the same honesty property (still explicitly imputed,
     still sourced from COCOMO-NASA's real data, still flagged via
     source_dataset) while removing the artificial value-count skew a
     single constant created.

4. source_dataset
   - A flag (0 = COCOMO-NASA, 1 = Desharnais, 2 = China — extended from
     binary when China was added, see decision #6 below), added as its
     own feature per the roadmap's suggested fallback option. Lets the
     unified model learn any systematic difference between the data
     collection contexts (including implicitly "this row's complexity
     and/or team_experience value is imputed, not measured") rather than
     the imputation in (3) silently masquerading as real signal. Kept as
     a plain integer (not one-hot) for consistency with how COCOMO's own
     binary `forg` column is already treated — every downstream use
     (scaling, SOFTWARE_SOURCE_DEFAULT in app.py, the "Historical Data
     Source" feature-importance label) reads this column generically
     (mode(), a plain int compare, a dataset-count-agnostic label) with
     no code that assumes exactly two values, verified when China was
     added.

5. effort_person_months (target)
   - COCOMO-NASA's `act_effort` is already person-months — used as is.
   - Desharnais's `Effort` is person-hours — converted to person-months
     using the same Boehm/COCOMO HOURS_PER_PERSON_MONTH=152 constant
     already used in app.py, now shared from preprocessing.py.
   - China's `Effort` (see decision #6) is also person-hours — same
     HOURS_PER_PERSON_MONTH conversion.

6. THIRD DATASET ADDED: China (PROMISE repository, data/raw/china.arff,
   499 rows) — see results/software_unification_notes.md, "Adding China
   (PROMISE) for small-project coverage", for the full investigation.
   Column mapping, verified against the actual downloaded file (not
   assumed from any prompt):
   - project_size_kloc: AFP (Adjusted Function Points) x
     BACKFIRE_LOC_PER_FP / 1000 — same Capers Jones backfire heuristic
     already used for Desharnais, for methodological consistency.
   - effort_person_months: Effort (person-hours) / HOURS_PER_PERSON_MONTH
     — NOT N_effort (see exclusions below).
   - EXCLUDED as features (verified, not assumed):
     - PDR_AFP, PDR_UFP, NPDR_AFP, NPDU_UFP: "productivity delivery
       rate" columns computed FROM the effort target. Confirmed directly
       against the raw data: PDR_AFP == Effort / AFP (correlation
       0.9999971, max abs diff 0.05 — rounding only) and NPDR_AFP ==
       N_effort / AFP identically; PDR_UFP/NPDU_UFP are the same ratios
       against unadjusted function points (corr 0.99 with their AFP-based
       counterparts). Same leakage category as the Team Size derivation
       already rejected for Desharnais — a feature built from the
       training target, not a real pre-project input.
     - N_effort: correlates 0.986 with Effort itself (vs. 0.22-0.26 for
       the PDR/NPDR columns above) — a "normalized effort" variant of the
       target, not an independent input. Effort (not N_effort) is used
       for the target per decision #5.
     - Duration: the REALIZED/actual project duration, not something
       knowable before a project starts — same treatment as Desharnais's
       `Length` column.
     - Dev.Type: verified constant (value 0) across all 499 rows —
       `value_counts()` returns exactly one value — carries zero signal,
       dropped.
   - team_experience: China's only people-related column, `Resource`
     (values 1-4), has NO documented meaning. Checked directly: weak,
     non-monotonic correlation with effort/size/duration (0.16-0.25,
     median-by-group not even ordered 1<2<3<4), and a web search for the
     original PROMISE/Kitchenham documentation turned up an explicit
     statement that this dataset "was received without further
     documentation details" — i.e. genuinely undocumented, not just
     undocumented-by-me. Rather than guess a semantic mapping onto the
     existing 0-5-ish team_experience scale, it's imputed the same
     honest way as complexity below: sampled from the REAL
     team_experience values actually observed in COCOMO-NASA + Desharnais
     combined (seeded, reproducible, not a single constant), flagged via
     source_dataset=2.
   - complexity: no analogous column, same treatment as Desharnais —
     sampled from COCOMO-NASA's own empirical complexity distribution,
     reusing the exact same distribution/pattern as Desharnais's
     imputation (a separate seeded RandomState so China's rows don't
     perturb Desharnais's already-documented draw).

Rows with missing values are handled exactly as Step 4 already
established per dataset (COCOMO-NASA: none; Desharnais: the 4 rows with
'?' in TeamExp/ManagerExp are dropped) — reusing
preprocessing.handle_missing_values, not reimplemented. China has zero
missing values in any column (verified directly), so no row-dropping or
imputation-by-median is needed there at all — only the two columns with
no analogous source (team_experience, complexity) are imputed, as above.
"""

from __future__ import annotations

import os

import numpy as np
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
    load_raw_arff,
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

# Seed for the proportional complexity-imputation draw (see decision #3
# above) — fixed so the unified dataset, and everything trained on it, is
# reproducible across reruns, same reasoning as split_data.RANDOM_STATE.
RNG_SEED = 42

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
    # in by the caller from COCOMO-NASA's own complexity DISTRIBUTION (not
    # a single constant — see decision #3's "REVISED" note), so the
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


def _load_china_rows() -> pd.DataFrame:
    raw_df = load_raw_arff(f"{RAW_DIR}/china.arff")
    # No handle_missing_values call needed — verified directly that
    # china.arff has zero missing values in any column (see decision #6).

    project_size_kloc = (raw_df["AFP"].astype(float) * BACKFIRE_LOC_PER_FP) / 1000.0
    effort_months = raw_df["Effort"].astype(float) / HOURS_PER_PERSON_MONTH

    # team_experience and complexity: both imputed by the caller (decision
    # #6 — Resource has no usable, documented meaning; China has no
    # complexity-like column at all), same pattern as Desharnais's
    # complexity imputation. PDR_AFP/PDR_UFP/NPDR_AFP/NPDU_UFP/N_effort
    # (leakage) and Duration/Dev.Type (not usable — see decision #6) are
    # simply never read from raw_df above, so they can't leak in by
    # accident.
    unified = pd.DataFrame(
        {
            TEAM_EXPERIENCE_COL: pd.NA,
            PROJECT_SIZE_COL: project_size_kloc,
            COMPLEXITY_COL: pd.NA,
            SOURCE_COL: 2,
            TARGET_COL: effort_months,
        }
    )
    return unified


def build_unified_dataframe() -> pd.DataFrame:
    cocomo_rows = _load_cocomo_rows()
    desharnais_rows = _load_desharnais_rows()
    china_rows = _load_china_rows()

    # COCOMO-NASA's own empirical complexity distribution — shared by both
    # imputations below (Desharnais and China each lack a complexity
    # column entirely; decisions #3 and #6).
    complexity_dist = cocomo_rows[COMPLEXITY_COL].value_counts(normalize=True)

    # Impute Desharnais's missing complexity by sampling from that
    # distribution — documented decision #3's "REVISED" note above. A
    # single constant (the old approach) would collapse every imputed row
    # onto one value, artificially starving the other complexity levels
    # of size-diverse examples; sampling proportionally to COCOMO-NASA's
    # real distribution keeps each level represented in roughly its
    # real-world proportion instead. Unchanged from before China was
    # added, so these specific values stay exactly as already documented.
    rng = np.random.RandomState(RNG_SEED)
    desharnais_rows[COMPLEXITY_COL] = rng.choice(
        complexity_dist.index, size=len(desharnais_rows), p=complexity_dist.to_numpy()
    )

    # China's missing complexity AND team_experience (decision #6) — its
    # own independently-seeded RandomState (same RNG_SEED, a separate
    # instance) so adding China doesn't perturb Desharnais's draw above;
    # still fully reproducible across reruns.
    rng_china = np.random.RandomState(RNG_SEED)
    china_rows[COMPLEXITY_COL] = rng_china.choice(
        complexity_dist.index, size=len(china_rows), p=complexity_dist.to_numpy()
    )
    # team_experience: sampled from the REAL (non-imputed) team_experience
    # values observed across COCOMO-NASA + Desharnais combined — not a
    # single constant, same honesty standard as the complexity imputation.
    real_team_experience = pd.concat(
        [cocomo_rows[TEAM_EXPERIENCE_COL], desharnais_rows[TEAM_EXPERIENCE_COL]], ignore_index=True
    )
    china_rows[TEAM_EXPERIENCE_COL] = rng_china.choice(real_team_experience.to_numpy(), size=len(china_rows))

    combined = pd.concat([cocomo_rows, desharnais_rows, china_rows], ignore_index=True)
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
    print(f"China rows:       {(raw_unified[SOURCE_COL] == 2).sum()}")
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
