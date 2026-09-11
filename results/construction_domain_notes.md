# EstimateX — Construction Domain Notes

## Source / citation

**UCI Residential Building Data Set.** Rafiei, M.H. & Adeli, H. (2018). "A Novel Machine Learning Model for Estimation of Sale Prices of Real Estate Units." *Journal of Construction Engineering and Management*, 144(9).

- Repository page: https://archive.ics.uci.edu/dataset/437/residential+building+data+set
- Download: https://archive.ics.uci.edu/static/public/437/residential+building+data+set.zip
- Original format: Excel (`Residential-Building-Data-Set.xlsx`), two sheets (`Data`, `Descriptions`)
- 372 real residential construction projects in Tehran, Iran, each with physical/financial project variables and time-lagged economic indicators
- Saved unmodified (format-flattened only, see below) as `data/raw/construction_residential.csv`

## What the raw file contains

109 columns, 372 rows, **zero missing values** anywhere (confirmed at conversion time). The source Excel file uses a 2-row merged header (a group label row + a variable-ID row); since CSV can't represent merged headers, the conversion flattened it into one unique header row per column — no data values, row order, or column meaning were changed, the exact same kind of lossless structural conversion already used for the ARFF→CSV conversion of COCOMO-NASA/Desharnais back in the original roadmap's dataset step.

Column groups, per the source's own `Descriptions` sheet:
- 4 columns: project start/completion year + quarter (Persian calendar)
- **V-1 to V-8**: "PROJECT PHYSICAL AND FINANCIAL VARIABLES" — locality code, floor area, lot area, three variants of preliminary cost estimate, planned construction duration, and starting unit price per m²
- **V-9, V-10**: the two columns the dataset's own creators explicitly label **"(output)"** — V-9 is actual sales price, V-10 is actual construction cost
- **V-11 to V-29, repeated once per each of 5 time lags (95 columns)**: macroeconomic indicators at the time of each lag — interest rates, price indices, loan volumes, exchange rates, population, gold price, etc.

## The duration finding — answered explicitly, as requested

**The dataset DOES have a duration-related column: V-7, "Duration of construction."** But it is important to be precise about what kind of column it is:

- V-7 is listed under "PROJECT PHYSICAL AND FINANCIAL VARIABLES" — an **input** variable, alongside floor area, lot area, and the preliminary cost estimates.
- Only **V-9 (sales price)** and **V-10 (construction cost)** are labeled `(output)` by the dataset's own creators. V-7 is never treated as something to be predicted — it's the project's *planned/estimated* duration, known before or during construction, used here (like floor area or lot area) as a predictor of the eventual cost.
- The dataset provides no "actual duration" outcome distinct from this planned figure — there is nothing to train or evaluate an actual-duration prediction against.

**Conclusion: this domain predicts COST ONLY, not time**, exactly as the fallback the roadmap specified for a no-duration case — even though a duration-shaped column exists, it isn't usable as a prediction target here. V-7 is still used as an input FEATURE (see below) since it's legitimately informative and available pre-construction; the trained model's own feature-importance ranking (`results/feature_importance_construction.png`) confirms this — `planned_duration` is the second most important predictor of cost, right behind the preliminary cost estimate. The web UI, if/when this domain is added to it, needs to say plainly that it predicts cost only for Construction, not time, rather than fabricate a duration estimate the underlying model was never trained to produce.

## Target and feature choices

- **Target: V-10, "Actual construction costs (output)"** — the direct analog to what EstimateX predicts elsewhere ("cost"). Reported in the dataset's own units, "10000 IRRm" (i.e. figures are in units of 10,000, Iranian Rial, per the source's own — not entirely unambiguous — unit label; reported verbatim rather than reinterpreted).
- **V-9 (actual sales price) was dropped entirely, not used as a feature.** It's a second dataset-provided "answer" highly correlated with construction cost — using it as a predictor would be a leakage risk, not a genuine pre-construction input.
- **Features: only V-1 to V-8** (locality, floor area, lot area, the three preliminary-cost variants, planned duration, starting unit price) — the "project physical and financial variables" group, i.e. attributes of the project itself.
- **The 95 economic-indicator columns (V-11–V-29 × 5 time lags) were excluded.** These are macroeconomic context (interest rates, price indices, gold price, population, ...), not project attributes. Including 95 mostly-collinear macro columns on only 372 rows would risk overfitting, and would abandon the project-attribute-level explainability framing used everywhere else in EstimateX — COCOMO-NASA's and Desharnais's features are all about the project being estimated, not the economy around it.
- **The 4 raw start/completion year+quarter columns were dropped** — V-7 (duration) is already the direct summary of that date range, so keeping both would be redundant, and the raw Persian-calendar year values are period-specific rather than a generalizable numeric signal.

## How this pipeline differs from the software one

Construction is a genuinely **independent** pipeline (`src/preprocess_construction.py`), not forced into the software domain's schema, per the roadmap's explicit instruction:

| | Software-Unified | Construction |
|---|---|---|
| Source | 2 harmonized datasets (COCOMO-NASA + Desharnais) | 1 dataset (UCI Residential Building) |
| Rows | 170 | 372 |
| Features | 4 (`team_experience`, `project_size_kloc`, `complexity`, `source_dataset`) | 27 (7 numeric + 20 one-hot `locality` columns) |
| Missing values | Desharnais's known 4 rows (handled per Step 4) | None found |
| Categorical encoding | Ordinal (`complexity`, preserving the vl–xh order) | One-hot (`locality` — a nominal zone code, no inherent order) |
| Target unit | Person-months (harmonized) | Native dataset units (10,000 IRR) |
| Missing-column handling | Explicit imputation (`complexity` for Desharnais) + `source_dataset` flag | Not needed — no missing columns |

Both pipelines reuse the same generic building blocks from `src/preprocessing.py` (`handle_missing_values`, `encode_onehot`, `scale_numeric_features`, `save_artifact`) and the same generic training/evaluation/selection functions from `src/train_models.py`, `src/evaluate.py`, and `src/select_final_model.py` (via `src/train_new_domains.py`) — only the feature engineering itself is domain-specific.

## Results

| Model | MAE | RMSE | MMRE | PRED(25) |
|---|---|---|---|---|
| Linear Regression | 23.96 | 32.38 | 0.151 | 80.0% |
| **Random Forest (selected)** | 26.91 | 43.73 | 0.136 | 82.7% |
| XGBoost | 28.36 | 49.48 | 0.124 | 86.7% |
| SVR | 122.08 | 157.98 | 0.844 | 22.7% |

By raw MAE+PRED(25) alone, Linear Regression would win — but it has no feature-importance support, and per Step 9's rule, explainability is a hard requirement, not optional. Among the tree-based models, Random Forest edges out XGBoost (MAE 26.91 vs. 28.36 — XGBoost's gap is 5.40%, just over the 5% tolerance used throughout this project, even though XGBoost has the better PRED(25) at 86.7% vs. 82.7%). **Random Forest was selected** — saved as `models/final_model_construction.pkl` with `models/final_scaler_construction.pkl` and `models/final_encoder_construction.pkl`.

Worth noting for the paper: Construction's PRED(25) (82.7%) is dramatically higher than either software domain's (see `results/model_comparison_new_domains.csv` and the original `results/model_comparison.csv`) and clears the ~75% "reliable estimator" benchmark referenced throughout this project's other write-ups — construction cost, driven overwhelmingly by a professional preliminary estimate that's already highly correlated with the actual outcome, is a substantially easier prediction problem than software effort estimation from qualitative project-attribute ratings.
