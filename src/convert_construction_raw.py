"""
EstimateX — one-time conversion of the UCI Residential Building Data Set
from its source Excel format into data/raw/construction_residential.csv.

Source: https://archive.ics.uci.edu/dataset/437/residential+building+data+set
(direct download: https://archive.ics.uci.edu/static/public/437/residential+building+data+set.zip)
— download and unzip to get `Residential-Building-Data-Set.xlsx`, then run
this script with SOURCE_XLSX pointed at it.

The source file has a 2-row merged Excel header (a group-label row, then a
variable-ID row), with the economic-indicator variables (V-11..V-29)
repeated once per each of 5 time lags. CSV can't represent merged headers,
so this flattens the header into one unique row per column, disambiguating
the repeated block with a `_lag` suffix (e.g. V-11_lag1..V-11_lag5) — a
structural flattening only, exactly like the ARFF->CSV conversion used for
COCOMO-NASA/Desharnais: no data values, row order, or column meaning are
altered, only the header made flat and machine-readable.

Requires `openpyxl` (only needed for this one conversion script — not a
runtime dependency of the rest of the pipeline, which reads the already-
converted CSV).
"""

import pandas as pd

SOURCE_XLSX = "Residential-Building-Data-Set.xlsx"  # path to the downloaded source file
DEST_CSV = "data/raw/construction_residential.csv"

# V-1..V-10 each appear once; V-11..V-29 repeat once per time lag.
NON_REPEATING_VARS = {f"V-{i}" for i in range(1, 11)}


def convert(source_xlsx: str = SOURCE_XLSX, dest_csv: str = DEST_CSV) -> pd.DataFrame:
    raw = pd.read_excel(source_xlsx, sheet_name="Data", header=None)
    var_row = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)

    cols = []
    lag_counters: dict[str, int] = {}
    for i in range(len(var_row)):
        vid = var_row[i]
        if i < 4 or vid in NON_REPEATING_VARS:
            cols.append(vid)
        else:
            lag_counters[vid] = lag_counters.get(vid, 0) + 1
            cols.append(f"{vid}_lag{lag_counters[vid]}")

    assert len(set(cols)) == len(cols), "column flattening produced duplicate names"
    data.columns = cols

    data.to_csv(dest_csv, index=False)
    return data


if __name__ == "__main__":
    df = convert()
    print(f"Saved -> {DEST_CSV}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
