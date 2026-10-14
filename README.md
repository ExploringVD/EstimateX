# EstimateX — Predicting Product Development Time and Cost Using Machine Learning

EstimateX is a machine learning system trained on historical project data that predicts software and construction project effort/cost across two independent project domains. Rather than acting as a black box, it explains its predictions through feature importance, and ships as a live, interactive Flask web app, not just a notebook.

- **Software** (Software/Web, Mobile App, Enterprise/Other) — a sign-constrained linear model trained directly on COCOMO-NASA's own 93 projects and all 15 Boehm-1981 effort-driver ratings (not an averaged multi-dataset blend), predicting effort in person-months. See "Software model" below for why and how.
- **Construction** — a Random Forest model trained on the UCI Residential Building Data Set, predicting construction cost (native "10,000 IRR" units). Unchanged throughout the software model's redesign below.

## Software model

The deployed software model is `log(effort) = log(CPLX multiplier) + Ridge(14 other COCOMO drivers as log(published multiplier), log(KLOC), development mode)`, fit on COCOMO-NASA's 93 rows. It replaced an earlier model trained on a 4-feature blend of COCOMO-NASA + Desharnais + China (kept on disk as `models/final_model_software_unified_baseline.pkl`, a documented baseline, not loaded by the app).

**Why a constrained model, not a plain regression.** A first attempt — plain `RidgeCV` on the 15 drivers' raw ordinal levels (vl=0..xh=5) — genuinely beat the unified baseline on CV PRED(25)/R²(log), but its Product Complexity (`cplx`) coefficient came out *negative*: picking "Very High" complexity predicted **less** effort than "Low," at every regularization strength tried (alpha 0.01 to 1000). This is real multicollinearity (cplx correlates with other drivers like time/stor/sced across only 93 rows), not a bug — a well-known failure mode when fitting raw linear regression to COCOMO-style data, since Ridge optimizes overall predictive accuracy, not each coefficient's individual sign.

**The fix:** every driver is encoded as `log(Boehm-1981 published effort multiplier)` for its rated level, not a raw 0-5 index — this already encodes each driver's correct theoretical direction (e.g. higher Analyst Capability's multiplier is *lower*, so `log(EM)` is already lower too). `cplx` is then pulled out of the fitted feature set entirely and added back as a **fixed, weight=1 offset** using Boehm's own published, monotonically increasing CPLX table (vl=0.70 ... xh=1.65) — so complexity's effect on the final prediction is guaranteed monotonic by construction, not by hoping regularization lands on a sensible sign. The remaining 14 drivers + log(KLOC) + development mode are fit with `Ridge(alpha=0.1, positive=True)` on the residual — `positive=True` is safe here specifically *because* `log(EM)` already carries the right sign, so forcing every weight non-negative keeps every driver's effect in its theoretically correct direction.

**Results — 10x repeated 10-fold CV (seeds 0-9), mean ± std:**

| Model | PRED(25) | R²(log) | MdMRE |
|---|---|---|---|
| Unified baseline (4-feature RF) | 22.5% ± 0.4 | 0.668 ± 0.004 | 0.540 ± 0.008 |
| **COCOMO sign-constrained Ridge (deployed)** | **48.9% ± 2.5** | **0.824 ± 0.008** | **0.256 ± 0.012** |

R²(raw) (computed on un-logged person-months) is notably unstable for the deployed model (mean around -1.9, driven by a handful of large-effort outliers once exponentiated back out of log-space) — PRED(25)/R²(log)/MdMRE, the metrics the deploy decision is actually based on, are all strong and stable; R²(raw) is reported here for transparency, not as a deploy criterion.

**Project category.** The results page also shows a Project category (Up to / Borderline around / Above 150 person-months — a 120–190 PM band around the reference point, wide on purpose since the regression has real error near that boundary). This is a threshold on the SAME predicted effort, not a separate classifier. Out-of-fold CV **category accuracy**: 91.7% ± 0.8% vs. a 61.3% majority-class baseline (`src/evaluate_software_category.py`) — always reported as "category accuracy," never as the regression model's own accuracy.

## Tech Stack

- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — preprocessing, Linear Regression, Random Forest, SVR
- **XGBoost** — gradient-boosted comparison model
- **Flask** — the deployed web interface (`app/`)
- **matplotlib / seaborn** — EDA and training-time comparison charts
- **joblib** — model/preprocessing artifact persistence

## Folder Structure

```
EstimateX/
├── data/
│   ├── raw/                        # Original datasets, unmodified
│   │   ├── cocomo_nasa.csv
│   │   ├── desharnais.csv
│   │   ├── china.arff
│   │   └── construction_residential.csv
│   └── processed/                  # Cleaned/encoded data and train/test splits for all domains
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
├── src/
│   ├── preprocessing.py            # Cleaning, encoding, scaling — shared building blocks
│   ├── preprocess_construction.py  # Construction domain preprocessing
│   ├── unify_software.py           # Merges COCOMO-NASA + Desharnais + China into one software dataset
│   ├── convert_construction_raw.py # One-time Excel -> CSV conversion for the construction dataset
│   ├── split_data.py               # Train/test split (random_state=42)
│   ├── train_models.py             # Trains + cross-validates all 4 algorithms per dataset
│   ├── evaluate.py                 # Test-set MAE/RMSE/MMRE/PRED(25) + comparison charts
│   ├── feature_importance.py       # Feature importance rankings + charts for the tree models
│   ├── select_final_model.py       # Picks and re-saves the final model + preprocessing artifacts
│   ├── train_new_domains.py        # Orchestrates the above for the construction domain (and the
│   │                                # now-superseded unified-baseline software model, still functional)
│   ├── tune_software_model.py      # Superseded intermediate attempt (log-target RF) — kept as a
│   │                                # documented step, not loaded by the app
│   ├── train_software_cocomo.py    # Trains + deploys the CURRENT software model (see "Software
│   │                                # model" above) — the only software training script the app's
│   │                                # models actually come from
│   └── evaluate_software_category.py  # Out-of-fold CV "project category" accuracy check
├── models/                         # final_model_software.pkl / final_model_software_metadata.pkl
│                                    # / final_model_construction.pkl / final_scaler_* / final_encoder_*
│                                    # — the artifacts the app actually loads at runtime, plus
│                                    # final_model_software_unified_baseline.pkl and _OLD.pkl, kept as
│                                    # documented prior baselines (not loaded by the app)
├── app/
│   ├── app.py                      # Flask app — real predictions from the final trained models
│   ├── requirements.txt            # Minimal deps to run just the app (no training/EDA tooling)
│   ├── templates/                  # form.html, results.html (Jinja2, extend base.html)
│   └── static/                     # style.css, app.js, kloc-presets.js, results.js, favicon.png
├── EstimateX.pptx                  # Slide deck
├── requirements.txt                # Full dependencies (EDA, training, and the app)
└── README.md
```

## Datasets

| Dataset | Domain | Records | Description |
|---|---|---|---|
| **COCOMO-NASA** | Software | 93 | NASA software projects (1971–1987) with size, effort-driver ratings, and actual development effort in person-months. |
| **Desharnais** | Software | 81 | Canadian software projects with team/manager experience and function-point-based size against actual effort in person-hours. |
| **China** | Software | 499 | PROMISE repository software projects, added for small-project (low-KLOC) coverage. |
| **UCI Residential Building** | Construction | 372 | Iranian residential construction projects with physical/financial variables and actual construction cost. |

COCOMO-NASA and Desharnais are sourced from the [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository) (ARFF, losslessly converted to CSV). China is read directly from its original ARFF file. The construction dataset is from the [UCI Residential Building Data Set](https://archive.ics.uci.edu/dataset/437/residential+building+data+set), converted from its source Excel workbook via `src/convert_construction_raw.py`.

## Team

| Name | Roll No. |
|---|---|
| Vaishnavi Dutt | 2400320101217 |
| Varnika Singh | 2400320101234 |
| Vandana Singh | 2400320101219 |

**Guide:** Ms. Divya Maheshwari

---

## Quick Start — just run the web app

This is enough to get a working, live prediction demo running; it doesn't need the EDA/training tooling.

```bash
git clone https://github.com/ExploringVD/EstimateX
cd EstimateX

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r app/requirements.txt

python app/app.py
```

Then open **http://127.0.0.1:8000/** in a browser, fill in the form, and click "Get estimate" — you'll get a real prediction from the deployed models plus a live, per-request explanation of what drove it, not a placeholder. (macOS users: if you ever re-run model training and hit an XGBoost import error, it needs the OpenMP runtime — `brew install libomp` — but this isn't required just to run the app.)

## Full Setup — reproduce the entire pipeline

To regenerate everything from raw data through to the final deployed models (`models/final_model_software.pkl` + `final_model_software_metadata.pkl`, and `models/final_model_construction.pkl`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/preprocess_construction.py   # construction raw -> data/processed/construction_*.csv
python src/train_new_domains.py         # trains, evaluates, and selects the final construction model
python src/train_software_cocomo.py     # trains, CV-evaluates, and deploys the current software model
python src/evaluate_software_category.py  # prints the "project category" accuracy check
```

Each script is idempotent and safe to re-run — outputs are overwritten deterministically (fixed `random_state=42` for model hyperparameters; `src/train_software_cocomo.py`'s own CV uses seeds 0-9, documented in its own module docstring).

`src/train_software_cocomo.py` only deploys its model if it beats the existing `models/final_model_software.pkl` on CV PRED(25) **and** CV R²(log) — otherwise it leaves the deployed model untouched and prints why. The model it would otherwise replace is also what `src/train_new_domains.py` still trains and evaluates (the unified-baseline approach, superseded but kept functional and documented — see "Software model" above), and `src/unify_software.py` (COCOMO-NASA + Desharnais + China -> `data/processed/software_unified*.csv`) is that pipeline's own data-prep step; running `train_new_domains.py` on its own regenerates that baseline without touching the deployed software model.

The original per-dataset pipeline (COCOMO-NASA and Desharnais trained and deployed separately, before either the unification or the current COCOMO-native model) is still present and functional — `src/preprocessing.py`, `src/train_models.py`, `src/evaluate.py`, `src/feature_importance.py`, and `src/select_final_model.py` — and its generic functions are what `train_new_domains.py` reuses internally. Running it standalone regenerates the legacy `cocomo`/`desharnais` artifacts, which the app no longer loads:

```bash
python src/preprocessing.py
python src/split_data.py
python src/train_models.py
python src/evaluate.py
python src/feature_importance.py
python src/select_final_model.py
```
