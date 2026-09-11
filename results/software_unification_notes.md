# EstimateX — Software Domain Unification Notes

Full column-mapping reasoning also lives as inline documentation in `src/unify_software.py` (required by the roadmap: "document every mapping decision in a comment"); this file mirrors it in one place for the paper, alongside the construction domain's equivalent notes.

## Why

Before this, "software" was two separately-trained, separately-deployed models (COCOMO-NASA, Desharnais) behind one UI label. This unifies them into a genuinely single software-effort dataset and model, so each of EstimateX's domains — software, construction — is actually one model, not a label hiding several.

## Investigated and confirmed absent: Team Size and Required Reliability

The web app originally showed "Team Size" and "Required Reliability" as form fields flagged "not yet used by the trained model" — a real explainability gap (a field shown to the user should genuinely feed the model, or not be shown at all). Re-investigated both, exhaustively, before deciding what to do:

- **Team Size (headcount)**: neither raw dataset has one. COCOMO-NASA's people-related columns (`acap`, `aexp`, `pcap`, `vexp`, `lexp`) are capability/experience **ratings** (vl–xh), not counts. Desharnais's (`TeamExp`, `ManagerExp`) are experience in **years**, not counts. Checked whether a count could be legitimately *derived* instead (e.g. from `Effort` / `Length` in Desharnais) — rejected, since that would use the training target (`Effort`) to build a feature, a direct leakage violation, not a real pre-project input.
- **Required Reliability**: already established in Step 10 as absent from Desharnais; re-confirmed here — still no analogous column.

**Decision: both fields were removed from the software form entirely**, applying the same rule the roadmap specified for Reliability to Team Size once it turned out to apply there too — showing a field that doesn't genuinely drive the prediction is worse than not showing it, even if the field "feels" like it should exist. The software form now asks only for what the unified model actually uses: Project Domain, Team Experience, Project Size (KLOC), Project Complexity.

**Proof nothing was missed**: re-ran `src/unify_software.py` and `src/train_new_domains.py` end-to-end after this investigation — `results/model_comparison_new_domains.csv` and `results/feature_importance_software.png` came back **byte-for-byte identical** to before, confirming the 4-feature schema (`team_experience`, `project_size_kloc`, `complexity`, `source_dataset`) was already complete and correct; there was no missing column to add.

## Unified schema

`team_experience`, `project_size_kloc`, `complexity`, `source_dataset`, target = `effort_person_months`. Chosen to match exactly what the roadmap specified (team size/experience, project size, complexity, target=effort) rather than a superset padded with COCOMO-only columns that would just be constant-imputed on every Desharnais row.

| Unified column | COCOMO-NASA source | Desharnais source |
|---|---|---|
| `team_experience` | Mean of 5 ordinal ratings (`acap`, `aexp`, `pcap`, `vexp`, `lexp`), ordinal-encoded 0–5 | Mean of `TeamExp` and `ManagerExp` (both native 0–4 years) |
| `project_size_kloc` | `equivphyskloc` directly (already KLOC) | `PointsAdjust` (function points) converted via Capers Jones' ~100 LOC/function-point "backfire" heuristic — the same constant already used in `app.py`'s live Desharnais prediction path, now shared from `preprocessing.py` |
| `complexity` | `cplx` directly (already the vl–xh scale) | **No analogous column exists.** Imputed with COCOMO-NASA's modal `cplx` value — a stated default, not a fabricated measurement |
| `source_dataset` | 0 | 1 — lets the model learn any systematic difference between the two collection contexts, including that Desharnais's `complexity` is imputed, not measured |
| `effort_person_months` (target) | `act_effort` directly (already person-months) | `Effort` (person-hours) ÷ 152 (Boehm/COCOMO's standard hours-per-person-month figure, already used in `app.py`, now shared from `preprocessing.py`) |

Missing values handled exactly as Step 4 already established per source dataset before unification (COCOMO-NASA: none; Desharnais: the 4 rows with `?` in `TeamExp`/`ManagerExp` dropped) — reused via `preprocessing.handle_missing_values`, not reimplemented.

**Result: 93 + 77 = 170 unified rows** (up from 93 or 77 individually) — a real, if modest, mitigation of the "under 200 records" limitation flagged throughout this project's other write-ups, simply by combining what were previously two separate small datasets into one.

## Validation the unification actually worked

`source_dataset`'s feature importance in the trained Random Forest is **0.005 — by far the least important of the 4 features** (`results/feature_importance_software.png`). That's a meaningful sanity check: if the model were mostly relying on "which original dataset did this row come from" to make predictions, unification would have papered over a real gap rather than closing it. Instead, `project_size_kloc` (0.65) and `complexity` (0.26) dominate — the model is predicting off harmonized project attributes, not off which source system a row originally came from.

## Results

| Model | MAE | RMSE | MMRE | PRED(25) |
|---|---|---|---|---|
| Linear Regression | 273.40 | 403.39 | 2.953 | 5.9% |
| **Random Forest (selected)** | 253.74 | 673.00 | 0.810 | 29.4% |
| XGBoost | 293.19 | 913.65 | 0.800 | 32.4% |
| SVR | 335.00 | 640.58 | 1.481 | 11.8% |

Random Forest wins on MAE outright (253.74, lowest of all 4) and is tree-based, so no explainability fallback was needed here (unlike Construction — see `results/construction_domain_notes.md`). XGBoost has a higher PRED(25) (32.4% vs. 29.4%), but its MAE is 15.55% higher than Random Forest's — well over the 5% tolerance used throughout this project — so Random Forest is kept.

**Honest comparison to the original per-dataset models**: this combined-domain PRED(25) of 29.4% sits between COCOMO-NASA's standalone 42.1% and Desharnais's standalone 25.0% (`results/model_comparison.csv`) — unification traded away some of COCOMO-NASA's stronger individual accuracy for a single, simpler, genuinely-one-model software domain with more total training data. That tradeoff is worth stating plainly in the paper rather than only presenting the upside.

Saved as `models/final_model_software.pkl`, `models/final_scaler_software.pkl`, `models/final_encoder_software.pkl` — per the roadmap, `models/final_model_cocomo.pkl` and `models/final_model_desharnais.pkl` are kept on disk (nothing deleted), simply no longer what a unified software domain would route to.
