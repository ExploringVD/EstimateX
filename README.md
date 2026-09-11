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
