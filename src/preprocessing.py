"""
EstimateX — Step 4: cleaning & preprocessing.

Reusable, importable building blocks for turning raw datasets into
model-ready feature matrices, plus two dataset-specific pipelines
(COCOMO-NASA, Desharnais) that apply them with the choices justified by
the EDA in notebooks/01_eda.ipynb. A third software dataset (China,
PROMISE repository) was added later — it reuses this module's generic
building blocks (load_raw_arff below, handle_missing_values, etc.) from
src/unify_software.py directly, rather than getting its own pipeline
function here, since it feeds straight into the unified software model
instead of being evaluated standalone the way COCOMO-NASA/Desharnais were.

This module is imported (not just run as a script) by later training code
and by the deployed app, so every transform function is written to be
callable again on a *single new row* at inference time: pass a previously
fitted scaler/encoder in and it only transforms; leave it out and it fits
a fresh one (which is what Step 5's train/test split will do — fit on the
training split only, then transform-only on the test split / live input).

Raw files under data/raw/ are only ever opened for reading — nothing in
this module writes to that directory.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# The six-point COCOMO effort-driver rating scale, low to high. Encoding
# these as ordinals (rather than one-hot) keeps their natural order, per
# the EDA notebook's summary finding.
COCOMO_ORDINAL_SCALE = ["vl", "l", "n", "h", "vh", "xh"]
COCOMO_ORDINAL_COLS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]
# Nominal (unordered) categoricals — one-hot encoded. `center` is a NASA
# facility identifier and `mode` a development-mode label; both are codes,
# not quantities, despite `center` being stored as an int.
COCOMO_ONEHOT_COLS = ["projectname", "cat2", "mode", "center"]
# Binary categorical — label-encoded (0/1) rather than one-hot, since a
# single 0/1 column already captures a two-category variable with no
# extra sparsity.
COCOMO_BINARY_COL = "forg"
# Continuous numeric columns, scaled directly.
COCOMO_NUMERIC_COLS = ["year", "equivphyskloc"]
# Pure row identifier — carries no predictive signal; the EDA notebook
# flagged its r≈0.39 correlation with the target as spurious (an artifact
# of rows being roughly ordered by year), so it's dropped as a predictor.
COCOMO_ID_COL = "recordnumber"
COCOMO_TARGET_COL = "act_effort"

# Desharnais: all predictors are already numeric except Language, which
# is a nominal category code (1/2/3) — the EDA notebook explicitly flagged
# it as "categorical, not ordinal" (its correlation sign doesn't follow the
# 1<2<3 ordering), so it's one-hot, not ordinal, encoded.
DESHARNAIS_ONEHOT_COLS = ["Language"]
DESHARNAIS_NUMERIC_COLS = [
    "TeamExp", "ManagerExp", "YearEnd", "Length",
    "Transactions", "Entities", "PointsAdjust", "Envergure", "PointsNonAjust",
]
# Pure row identifier — dropped as a predictor for the same reason as
# COCOMO's recordnumber.
DESHARNAIS_ID_COL = "Project"
DESHARNAIS_TARGET_COL = "Effort"

# Shared unit-conversion constants — used wherever software effort/size
# figures need to cross between COCOMO-NASA's and Desharnais's different
# units (the live app's Desharnais path, and the software-domain
# unification pipeline). Defined once here rather than redefined per
# call site.
#
# Boehm/COCOMO's standard person-hours-per-person-month figure, used to
# convert Desharnais's effort (person-hours) into the person-months unit
# COCOMO-NASA's effort is already in.
HOURS_PER_PERSON_MONTH = 152

# Capers Jones' commonly-cited "backfire" rule of thumb: roughly 100 lines
# of code per function point for typical 3rd-generation languages. Used to
# give Desharnais's function-point-based size measures an approximate
# KLOC-equivalent, comparable to COCOMO-NASA's native `equivphyskloc`. A
# named, documented approximation — not a measured conversion.
BACKFIRE_LOC_PER_FP = 100


# ---------------------------------------------------------------------------
# Generic, reusable building blocks
# ---------------------------------------------------------------------------

def load_raw_csv(path: str, na_placeholders=("?",)) -> pd.DataFrame:
    """Read a raw CSV, treating known placeholder tokens (e.g. the literal
    '?' the EDA found in Desharnais) as missing. Read-only — never writes
    back to `path`.
    """
    df = pd.read_csv(path)
    if na_placeholders:
        df = df.replace(list(na_placeholders), np.nan)
    return df


def load_raw_arff(path: str, na_placeholders=("?",)) -> pd.DataFrame:
    """Read a raw ARFF file's @data section into a DataFrame, using the
    file's own @attribute list for column names/order. Placed here rather
    than in its own module (e.g. src/load_china.py) because it's a
    generic building block in the same spirit as load_raw_csv above —
    the dataset-specific column-mapping decisions for whatever's loaded
    with it belong in that dataset's own unify_*.py, not here.

    Deliberately NOT a general ARFF parser — it only handles what the
    China (PROMISE) dataset's file actually uses: a flat @relation, plain
    `@attribute name numeric` declarations (no {nominal,...} categories,
    no sparse `{idx value, ...}` data rows, no quoted strings), and a
    single comma-separated @data section. Good enough for this file
    without pulling in an external `arff` dependency; would need
    extending (or swapping for a real library) for a fancier ARFF file.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    attribute_names = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lower = stripped.lower()
        if lower.startswith("@attribute"):
            # "@attribute NAME TYPE" — NAME is whatever token comes right
            # after @attribute; TYPE (here always "numeric") is unused.
            attribute_names.append(stripped.split()[1])
        elif lower.startswith("@data"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"{path}: no @data section found")
    if not attribute_names:
        raise ValueError(f"{path}: no @attribute declarations found")

    data_rows = [line.split(",") for line in lines[data_start:] if line.strip()]
    df = pd.DataFrame(data_rows, columns=attribute_names)
    if na_placeholders:
        df = df.replace(list(na_placeholders), np.nan)
    # ARFF's "numeric" type covers both ints and floats; every column here
    # is read back as text by the split() above, so coerce uniformly —
    # same reasoning as handle_missing_values' numeric_cols coercion.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    categorical_cols: Optional[list[str]] = None,
    drop_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Column-by-column missing-value handling, driven by how much of the
    dataset a column's missing values actually represent:

    - missing fraction <= drop_threshold: drop just those rows. Losing a
      handful of rows out of the whole dataset is cheap, and it avoids
      inventing values for a column where we barely have any evidence of
      what's missing.
    - missing fraction > drop_threshold: impute instead (median for
      numeric columns, mode for categorical), since dropping would throw
      away too much of the dataset.

    Returns (cleaned_df, report) where report maps column -> details of
    what was done, for the printed summary.
    """
    df = df.copy()
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    # Coerce numeric columns to actual numeric dtype first — a column with
    # even one placeholder (e.g. Desharnais' TeamExp) is read as text/object,
    # which would otherwise silently break a median calculation.
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_rows = len(df)
    report: dict = {}
    rows_to_drop: set = set()

    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            continue

        missing_fraction = missing_count / n_rows
        if missing_fraction <= drop_threshold:
            rows_to_drop.update(df.index[missing_mask].tolist())
            report[col] = {
                "missing_count": missing_count,
                "missing_fraction": round(missing_fraction, 4),
                "action": "drop_rows",
            }
        elif col in numeric_cols:
            fill_value = float(df[col].median())
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "missing_count": missing_count,
                "missing_fraction": round(missing_fraction, 4),
                "action": "impute_median",
                "fill_value": fill_value,
            }
        else:
            fill_value = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "missing_count": missing_count,
                "missing_fraction": round(missing_fraction, 4),
                "action": "impute_mode",
                "fill_value": fill_value,
            }

    n_dropped = len(rows_to_drop)
    if rows_to_drop:
        df = df.drop(index=list(rows_to_drop))
    df = df.reset_index(drop=True)

    report["_rows_dropped_total"] = n_dropped
    return df, report


def scale_numeric_features(
    df: pd.DataFrame,
    columns: list[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize `columns` in place (mean 0, std 1).

    Pass `scaler=None` to fit a fresh StandardScaler on `df` (what this
    module does now, fitting on the full cleaned dataset since there's no
    train/test split yet). Pass an already-fitted scaler to transform-only
    — that's how Step 5 will reuse this same function: fit on the training
    split, then transform the test split / a live app input with the
    scaler saved here, without refitting.
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler


def encode_onehot(
    df: pd.DataFrame,
    columns: list[str],
    encoder: Optional[OneHotEncoder] = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode `columns`, replacing them with dummy columns.

    `handle_unknown="ignore"` so a category never seen during fitting (e.g.
    a new project type submitted through the deployed app) produces an
    all-zero row instead of raising.
    """
    df = df.copy()
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=columns), encoded_df], axis=1)
    return df, encoder


def encode_ordinal(
    df: pd.DataFrame,
    columns: list[str],
    categories_order: list[str],
    encoder: Optional[OrdinalEncoder] = None,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode `columns` that all share the same category ordering
    (e.g. COCOMO's vl < l < n < h < vh < xh rating scale), in place.
    """
    df = df.copy()
    if encoder is None:
        encoder = OrdinalEncoder(categories=[categories_order] * len(columns))
        df[columns] = encoder.fit_transform(df[columns])
    else:
        df[columns] = encoder.transform(df[columns])
    return df, encoder


def encode_label_binary(
    df: pd.DataFrame,
    column: str,
    encoder: Optional[LabelEncoder] = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode a single binary categorical column to 0/1, in place."""
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
    else:
        df[column] = encoder.transform(df[column])
    return df, encoder


def save_artifact(obj, path: str) -> None:
    """joblib-dump a fitted scaler/encoder so the deployed app can apply
    the exact same transform to new input later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Dataset-specific pipelines
# ---------------------------------------------------------------------------

def preprocess_cocomo_nasa(
    raw_path: str = f"{RAW_DIR}/cocomo_nasa.csv",
    processed_path: str = f"{PROCESSED_DIR}/cocomo_nasa_clean.csv",
    models_dir: str = MODELS_DIR,
) -> dict:
    """Clean, encode, and scale the COCOMO-NASA dataset. Fits fresh
    scaler/encoders on the full cleaned dataset (no train/test split yet —
    that's Step 5) and saves them to `models_dir` for reuse.
    """
    raw_df = load_raw_csv(raw_path)
    raw_shape = raw_df.shape

    df = raw_df.drop(columns=[COCOMO_ID_COL])

    # The EDA notebook found zero missing values in COCOMO-NASA, so this
    # is expected to be a no-op — run it anyway so the pipeline stays
    # correct if that ever changes (e.g. a future data refresh).
    categorical_cols = COCOMO_ONEHOT_COLS + COCOMO_ORDINAL_COLS + [COCOMO_BINARY_COL]
    df, missing_report = handle_missing_values(
        df, numeric_cols=COCOMO_NUMERIC_COLS, categorical_cols=categorical_cols
    )

    df, ordinal_encoder = encode_ordinal(df, COCOMO_ORDINAL_COLS, COCOMO_ORDINAL_SCALE)
    df, binary_encoder = encode_label_binary(df, COCOMO_BINARY_COL)
    df, onehot_encoder = encode_onehot(df, COCOMO_ONEHOT_COLS)

    # Scale every genuinely numeric column: the two continuous features
    # plus the now-numeric (0-5) ordinal-encoded driver ratings. The
    # one-hot dummy columns and the binary forg column are left unscaled,
    # since scaling a 0/1 indicator has no meaning.
    scale_cols = COCOMO_NUMERIC_COLS + COCOMO_ORDINAL_COLS
    df, scaler = scale_numeric_features(df, scale_cols)

    # Move the target to the end and leave it completely untouched by any
    # of the transforms above.
    target = df.pop(COCOMO_TARGET_COL)
    df[COCOMO_TARGET_COL] = target

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    save_artifact(scaler, f"{models_dir}/cocomo_scaler.joblib")
    save_artifact(ordinal_encoder, f"{models_dir}/cocomo_ordinal_encoder.joblib")
    save_artifact(onehot_encoder, f"{models_dir}/cocomo_onehot_encoder.joblib")
    save_artifact(binary_encoder, f"{models_dir}/cocomo_forg_label_encoder.joblib")

    return {
        "dataset": "COCOMO-NASA",
        "raw_shape": raw_shape,
        "clean_shape": df.shape,
        "rows_dropped": missing_report.pop("_rows_dropped_total"),
        "missing_value_report": missing_report,
        "dropped_columns": [COCOMO_ID_COL],
        "target_column": COCOMO_TARGET_COL,
    }


def preprocess_desharnais(
    raw_path: str = f"{RAW_DIR}/desharnais.csv",
    processed_path: str = f"{PROCESSED_DIR}/desharnais_clean.csv",
    models_dir: str = MODELS_DIR,
) -> dict:
    """Clean, encode, and scale the Desharnais dataset. Fits fresh
    scaler/encoder on the full cleaned dataset (no train/test split yet —
    that's Step 5) and saves them to `models_dir` for reuse.
    """
    raw_df = load_raw_csv(raw_path)  # converts the literal '?' placeholders to NaN
    raw_shape = raw_df.shape

    df = raw_df.drop(columns=[DESHARNAIS_ID_COL])

    # TeamExp/ManagerExp: the EDA notebook found only 2 and 3 missing
    # values respectively out of 81 rows (<5%) — few enough that dropping
    # those rows (4 unique rows total) is cheaper than inventing values,
    # per the "drop when very few, impute when many" rule.
    df, missing_report = handle_missing_values(
        df,
        numeric_cols=DESHARNAIS_NUMERIC_COLS,
        categorical_cols=DESHARNAIS_ONEHOT_COLS,
    )

    df, onehot_encoder = encode_onehot(df, DESHARNAIS_ONEHOT_COLS)
    df, scaler = scale_numeric_features(df, DESHARNAIS_NUMERIC_COLS)

    target = df.pop(DESHARNAIS_TARGET_COL)
    df[DESHARNAIS_TARGET_COL] = target

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    save_artifact(scaler, f"{models_dir}/desharnais_scaler.joblib")
    save_artifact(onehot_encoder, f"{models_dir}/desharnais_onehot_encoder.joblib")

    return {
        "dataset": "Desharnais",
        "raw_shape": raw_shape,
        "clean_shape": df.shape,
        "rows_dropped": missing_report.pop("_rows_dropped_total"),
        "missing_value_report": missing_report,
        "dropped_columns": [DESHARNAIS_ID_COL],
        "target_column": DESHARNAIS_TARGET_COL,
    }


def _print_summary(summary: dict) -> None:
    print("=" * 70)
    print(summary["dataset"])
    print("-" * 70)
    print(f"Raw shape:   {summary['raw_shape']}")
    print(f"Clean shape: {summary['clean_shape']}")
    print(f"Rows dropped (missing values): {summary['rows_dropped']}")
    print(f"Columns dropped (non-predictive ID): {summary['dropped_columns']}")
    if summary["missing_value_report"]:
        print("Missing-value handling:")
        for col, detail in summary["missing_value_report"].items():
            print(f"  - {col}: {detail}")
    else:
        print("Missing-value handling: none needed (no missing values found)")
    print(f"Target column (left unscaled/unencoded): {summary['target_column']}")
    print()


if __name__ == "__main__":
    cocomo_summary = preprocess_cocomo_nasa()
    desharnais_summary = preprocess_desharnais()

    print()
    print("EstimateX — Step 4 preprocessing summary")
    _print_summary(cocomo_summary)
    _print_summary(desharnais_summary)
