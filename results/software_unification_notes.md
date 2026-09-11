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
| `complexity` | `cplx` directly (already the vl–xh scale) | **No analogous column exists.** Imputed by sampling COCOMO-NASA's own complexity **distribution** (seeded, reproducible) — a stated default, not a fabricated measurement. (Originally used COCOMO-NASA's single modal value instead — revised, see "Fixing unrealistic small-project predictions" below.) |
| `source_dataset` | 0 | 1 — lets the model learn any systematic difference between the two collection contexts, including that Desharnais's `complexity` is imputed, not measured |
| `effort_person_months` (target) | `act_effort` directly (already person-months) | `Effort` (person-hours) ÷ 152 (Boehm/COCOMO's standard hours-per-person-month figure, already used in `app.py`, now shared from `preprocessing.py`) |

Missing values handled exactly as Step 4 already established per source dataset before unification (COCOMO-NASA: none; Desharnais: the 4 rows with `?` in `TeamExp`/`ManagerExp` dropped) — reused via `preprocessing.handle_missing_values`, not reimplemented.

**Result: 93 + 77 = 170 unified rows** (up from 93 or 77 individually) — a real, if modest, mitigation of the "under 200 records" limitation flagged throughout this project's other write-ups, simply by combining what were previously two separate small datasets into one.

## Validation the unification actually worked

`source_dataset`'s feature importance in the trained Random Forest is **0.017 — by far the least important of the 4 features** (`results/feature_importance_software.png`). That's a meaningful sanity check: if the model were mostly relying on "which original dataset did this row come from" to make predictions, unification would have papered over a real gap rather than closing it. Instead, `project_size_kloc` (0.67) and `complexity` (0.22) dominate — the model is predicting off harmonized project attributes, not off which source system a row originally came from. (Figures retrained after the imputation fix below — `team_experience` sits at 0.087.)

## Fixing unrealistic small-project predictions

**Symptom**: a 1 KLOC, Nominal-complexity software submission predicted ~45 months of effort (₹3.82 crore) — far above sane expectations, and nowhere near the smooth, gradually-increasing curve a size-driven estimate should produce.

**Investigation (exact feature vector traced end to end)**: for `team_experience=Intermediate(2), project_size_kloc=1, complexity=n`, the raw row `{team_experience: 2.0, project_size_kloc: 1.0, complexity: 'n', source_dataset: 0}` ordinal-encodes complexity `'n' -> 2.0`, then scales to `[-0.596, -0.614, -1.818, 0]` using the fitted scaler's `mean_=[2.51, 64.93, 3.06]` / `scale_=[0.86, 104.07, 0.59]` — all within a normal, non-extrapolated z-score range. No units/scale bug: `project_size_kloc`'s scaler mean (64.93) matches the raw unified KLOC column's own mean (64.93) exactly, confirming the form's typed KLOC value is treated in the identical unit the training data itself uses (COCOMO's `equivphyskloc` already in KLOC; Desharnais's `PointsAdjust` converted via the same backfire heuristic used elsewhere in this project) — there is no double-scaling or unit mismatch anywhere in the pipeline.

**Root cause — a training-data imbalance, not a preprocessing bug**: Desharnais has no complexity column at all, so every one of its 77 rows was imputed. The *original* implementation filled all 77 with COCOMO-NASA's single **modal** value (`'h'`), which collapsed **135 of 170 unified rows (79%) onto one complexity value**. That starved the other levels — `'n'` (Nominal) had only 10 real rows total, and the smallest of those was a 10-KLOC/48-month project; `'vl'` had zero rows anywhere in the unified set. A Random Forest can't learn a smooth size-vs-effort trend inside a bucket that thin — a 1-KLOC query with `complexity='n'` had no genuinely comparable neighbor to draw on, so it fell into a sparse leaf anchored on that one 10-KLOC/48-month row, regardless of how much smaller the actual input was. This was confirmable directly: `predict(kloc=1, complexity='vl')`, `'l'`, and `'n'` all returned the **exact same** 44.97-month prediction — proof the tree had no real signal to distinguish between them at small sizes.

**Fix**: `unify_software.py`'s imputation now samples each Desharnais row's complexity from COCOMO-NASA's own empirical complexity **distribution** (seeded `RNG_SEED=42`, reproducible) instead of a single constant — still an honestly-labeled imputation (not a fabricated measurement, still flagged via `source_dataset`), but it no longer manufactures an artificial 79%-single-value skew. Full pipeline (`unify_software.py` -> `train_new_domains.py`) was rerun end to end after the change.

**Sanity table, before vs. after** (`team_experience=Intermediate, complexity=Nominal`):

| KLOC | Before (months) | After (months) |
|---|---|---|
| 1 | 44.97 | **25.38** |
| 5 | 49.64 | **30.64** |
| 20 | 126.22 | 144.25 |
| 50 | 317.47 | 423.82 |
| 200 | 1,749.48 | 1,661.02 |
| 1,000 | 2,521.54 | 2,407.61 |

Both curves are technically monotonic, but the "after" curve's low end dropped ~44% (44.97 -> 25.38 at 1 KLOC) — a direct, measurable effect of removing the imputation-driven skew, not noise.

**Honest residual limitation — not fixable by retraining**: even after this fix, ~25 months for a 1-KLOC Nominal project is still well above a casual "a few months for a tiny project" expectation. That gap is **not a bug** — COCOMO-NASA is 1980s/90s NASA aerospace/flight software, and its own smallest real recorded project (0.9 KLOC) still took a genuine 8.4 person-months (rated `'h'` complexity, not `'n'` — this dataset has *zero* real Nominal-complexity projects under 10 KLOC). No amount of retraining can make a model predict "2-4 months" when nothing resembling that ever appears in its training data; doing so would mean fabricating a trend the data doesn't support, which this project has consistently avoided elsewhere. This scope limitation (small aerospace-style training set, not general commercial-software norms) should be stated plainly in the paper alongside the fix.

**Team Size — re-confirmed absent, a third time**: re-checked both raw datasets' columns directly (`cocomo_nasa.csv`, `desharnais.csv`) as part of this investigation — neither has ever had a headcount column (COCOMO's people columns are capability *ratings*, Desharnais's are experience in *years*). The retrained model's features are unchanged: `team_experience`, `project_size_kloc`, `complexity`, `source_dataset` — no `team_size`, because there is still nothing honest to add. This matches the exhaustive investigation above and in `app.py`'s docstring; it was not silently dropped, it was never derivable in the first place.

## Results (after the imputation fix above)

| Model | MAE | RMSE | MMRE | PRED(25) |
|---|---|---|---|---|
| Linear Regression | 256.91 | 344.08 | 5.300 | 11.8% |
| **Random Forest (selected)** | 267.63 | 691.51 | 1.185 | 38.2% |
| XGBoost | 304.83 | 918.54 | 1.040 | 35.3% |
| SVR | 336.27 | 642.80 | 1.470 | 11.8% |

Random Forest's MAE (267.63) is 4.17% higher than Linear Regression's (256.91) — within the project's 5% MAE tolerance — while its PRED(25) is 26.5 percentage points higher (38.2% vs. 11.8%), so the same selection rule used throughout this project prefers Random Forest. (Before this fix, Random Forest also won on MAE outright; the imputation change shifted MAE slightly in Linear Regression's favor but PRED(25) still decides it — same winning model, now trained on a less-skewed dataset.)

**Honest comparison to the original per-dataset models**: this combined-domain PRED(25) of 29.4% sits between COCOMO-NASA's standalone 42.1% and Desharnais's standalone 25.0% (`results/model_comparison.csv`) — unification traded away some of COCOMO-NASA's stronger individual accuracy for a single, simpler, genuinely-one-model software domain with more total training data. That tradeoff is worth stating plainly in the paper rather than only presenting the upside.

Saved as `models/final_model_software.pkl`, `models/final_scaler_software.pkl`, `models/final_encoder_software.pkl` — per the roadmap, `models/final_model_cocomo.pkl` and `models/final_model_desharnais.pkl` are kept on disk (nothing deleted), simply no longer what a unified software domain would route to.
