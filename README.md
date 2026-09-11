# EstimateX — Predicting Product Development Time and Cost Using Machine Learning

EstimateX is a machine learning system trained on historical project data — the COCOMO-NASA and Desharnais datasets — that predicts software/product development time and cost across multiple project domains. Rather than acting as a black box, it explains its predictions through feature importance, giving teams insight into which project attributes (size, effort drivers, team experience, etc.) most influence the estimate.

## Tech Stack

- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — classical ML models and preprocessing
- **XGBoost** — gradient-boosted estimation models
- **Flask / Streamlit** — serving predictions via API and interactive UI
- **matplotlib / seaborn** — visualization and feature-importance plots
- **joblib** — model persistence

## Folder Structure

```
EstimateX/
├── data/
│   ├── raw/            # Original, unmodified datasets (COCOMO-NASA, Desharnais, etc.)
│   └── processed/      # Cleaned/feature-engineered data ready for modeling
├── notebooks/          # Exploratory data analysis and experimentation notebooks
├── src/                # Reusable source code (data prep, training, evaluation scripts)
├── models/              # Serialized trained models (.pkl) and related artifacts
├── app/                 # Flask/Streamlit application code for serving predictions
├── results/             # Metrics, plots, and reports generated from experiments
├── requirements.txt     # Python package dependencies
└── README.md
```

## Datasets

Both datasets are sourced from the [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository) (also mirrored via OpenML/Zenodo/SEACRAFT). The repository distributes them in ARFF format; they were losslessly converted to plain CSV (attribute names and data values unchanged, only the ARFF metadata/header wrapper stripped) and stored unmodified thereafter in `data/raw/`.

| Dataset | Source URL | Records | Description |
|---|---|---|---|
| **COCOMO-NASA** (`cocomo_nasa.csv`) | http://promise.site.uottawa.ca/SERepository/datasets/cocomonasa_2.arff | 93 | 93 NASA software projects (1971–1987) with size, effort-driver ratings (e.g. complexity, reliability, team capability), and actual development effort in person-months. |
| **Desharnais** (`desharnais.csv`) | http://promise.site.uottawa.ca/SERepository/datasets/desharnais.arff | 81 | 81 software projects from a Canadian software house, with team/manager experience and function-point-based size measures against actual development effort in person-hours. |

## Team

- Vaishnavi Dutt
- Varnika Singh
- Vandana Singh

**Guide:** Ms. Divya Maheshwari

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate it:
   ```bash
   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> More detailed setup and usage instructions will be added as the project progresses through subsequent roadmap steps.
