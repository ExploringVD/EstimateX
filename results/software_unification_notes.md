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

## Making the residual limitation visible to the end user (2026-09-11)

The "Honest residual limitation" above was previously only documented here — the deployed app itself presented every prediction with the same flat confidence, so a genuinely out-of-range query (e.g. 1 KLOC, Nominal) looked identical, from the UI's perspective, to a query the model actually has real support for. That's what was misread as "the model is broken."

`app/app.py` now computes `SOFTWARE_COMPLEXITY_COVERAGE` at startup — for each complexity level, the real count of training rows and (when count > 0) the smallest real KLOC/effort example on record, straight from `unify_software.build_unified_dataframe()` (raw, pre-scaling units, so it's an apples-to-apples comparison with the form's own KLOC input). `predict_software()` compares each request's KLOC against that level's real minimum and — only when the request falls outside real training coverage — attaches a plain-language `confidence_note`, rendered on the results page as a distinct caution block (`app/templates/results.html`, `.result-note-caution` in `app/static/style.css`) naming the actual closest real example. Requests within the training data's real range are unaffected — no note, same prediction as before.

This doesn't change any model, scaler, or prediction number — it makes an already-true, already-documented limitation visible where the person reading the estimate can actually see it, instead of leaving it buried in this file.

## Adding China (PROMISE) for small-project coverage

The "Honest residual limitation" section above concluded that COCOMO-NASA/Desharnais have **zero** real Nominal-complexity project under 10 KLOC, so small-project queries in that complexity band were extrapolating with no real support — a genuine data-coverage gap, not fixable by retraining on the same two datasets. This adds a third real dataset — [China (PROMISE repository)](https://github.com/Derek-Jones/Software-estimation-datasets/blob/master/china.arff), 499 software projects — to actually close that gap with real small projects instead of just flagging it.

### Loading the raw file

`data/raw/china.arff` is a standard ARFF file (verified directly: flat `@relation`, plain `@attribute name numeric` declarations, one comma-separated `@data` row per project, 499 rows, 19 columns — matches the file's own header exactly). Rather than add an external `arff` parsing dependency, `preprocessing.load_raw_arff()` (new, next to `load_raw_csv`) parses the `@attribute` list for column names/order and reads `@data` directly — deliberately not a general ARFF parser (documented in its own docstring: no nominal `{...}` categories, no sparse rows, no quoted strings), just enough for this file. Kept in `preprocessing.py` rather than a separate `src/load_china.py` for the same reason `load_raw_csv` lives there: it's a generic loading building block, and China's own column-mapping decisions belong in `unify_software.py` alongside COCOMO-NASA's and Desharnais's, not in a parallel module. The file has **zero missing values in any column** (verified directly), so no row-dropping or median-imputation was needed for any of its genuinely-present columns.

### Column mapping decisions (verified against the actual downloaded data, not assumed)

- **`project_size_kloc`**: `AFP` (Adjusted Function Points) × the same `BACKFIRE_LOC_PER_FP=100` Capers Jones heuristic already used for Desharnais, ÷ 1000 — identical treatment, for consistency.
- **`effort_person_months`**: `Effort` (person-hours, confirmed range 26-54,620) ÷ `HOURS_PER_PERSON_MONTH=152` — **not** `N_effort`, per below.
- **Excluded as leakage — confirmed, not assumed**:
  - `PDR_AFP`, `PDR_UFP`, `NPDR_AFP`, `NPDU_UFP`: checked directly against the raw data. `PDR_AFP == Effort / AFP` — correlation **0.9999971**, max absolute difference **0.05** (rounding only). `NPDR_AFP == N_effort / AFP` identically. `PDR_UFP`/`NPDU_UFP` correlate 0.99 with their AFP-based counterparts (same ratios against unadjusted function points). These are the training target divided by a size measure — the exact same leakage category as the Team Size-via-`Effort`/`Length` derivation already rejected for Desharnais.
  - `N_effort`: correlates **0.9863** with `Effort` itself (vs. 0.22-0.26 for the PDR/NPDR columns above) — clearly a variant/normalization of the target, not an independent input.
  - `Duration`: the realized/actual project duration — not knowable before a project starts, same treatment as Desharnais's `Length`.
  - `Dev.Type`: verified constant — `df['Dev.Type'].unique()` returns exactly `[0]` across all 499 rows. Zero signal, dropped.
- **`team_experience`**: China's only people-related column is `Resource` (values 1-4, confirmed via `unique()`). Checked whether it's a usable team-size/experience proxy before treating it as one:
  - Correlation with effort/size/duration is weak (0.16-0.25) and **not even monotonic** by group — median effort by `Resource` group is 1568 / 2361 / 4126 / 3344 for groups 1/2/3/4 respectively (group 4's median is *lower* than group 3's), which is not what a genuine ordinal team-size or experience measure would look like.
  - Searched for the original PROMISE/Kitchenham documentation for this specific column (multiple queries, checked the Zenodo record page, several arXiv papers on software effort datasets) — found no source that defines what `Resource`'s 1-4 values represent. This is a genuinely undocumented column, not something this project failed to find.
  - Given both the weak/non-monotonic signal and the absence of any documented meaning, `Resource` is **not** used as `team_experience`. Instead, `team_experience` is imputed the same honest way `complexity` already is for Desharnais: sampled from the REAL `team_experience` values observed in COCOMO-NASA + Desharnais combined (seeded `RNG_SEED=42`, a dedicated `RandomState` instance so this draw doesn't disturb Desharnais's existing complexity draw), not a single constant. Flagged via `source_dataset=2`, same honesty pattern as every other imputed value in this pipeline.
- **`complexity`**: no analogous column, same treatment as Desharnais — sampled from COCOMO-NASA's own empirical complexity distribution, **reusing the exact same `complexity_dist` computed in `build_unified_dataframe()`** (not a separate/reimplemented distribution) via its own seeded `RandomState`.
- **`source_dataset`**: extended from a binary 0/1 flag to `0=COCOMO-NASA, 1=Desharnais, 2=China`. Checked every downstream use before assuming this "just works": `app.py`'s `SOFTWARE_SOURCE_DEFAULT` uses `.mode()` (adapts automatically — now resolves to `2`, since China's 499 rows outnumber the other two combined), the "Historical Data Source" feature label/explanation text is dataset-count-agnostic (never said "two"), and the column is scaled/encoded nowhere as a binary-specific case (kept as a plain unscaled integer, same treatment as COCOMO's `forg` column, consistent with the existing 2-value design — not one-hot, since this project's tree models handle a small integer-coded categorical fine and the point of this column is only ever "which collection context," not an ordinal quantity). No code changes were needed anywhere outside `unify_software.py` for this to keep working correctly with three values.

### Retraining (unify_software.py → train_new_domains.py, full pipeline rerun)

**93 (COCOMO-NASA) + 77 (Desharnais) + 499 (China) = 669 unified rows** (up from 170) — a real, substantial mitigation of the "under 200 records" limitation, not just a modest one this time.

| Model | MAE | RMSE | MMRE | PRED(25) |
|---|---|---|---|---|
| Linear Regression | 152.96 | 243.44 | 8.486 | 7.5% |
| **Random Forest (selected)** | 130.84 | 419.88 | 2.165 | 20.1% |
| XGBoost | 140.41 | 693.53 | 2.108 | 19.4% |
| SVR | 106.84 | 315.98 | 1.166 | 15.7% |

SVR has the lowest MAE (106.84), but — same as every prior selection in this project — it isn't tree-based, so it has no feature-importance support and is disqualified by the explainability hard requirement. Among tree-based models, Random Forest wins outright (both lowest MAE and highest PRED(25) of the tree-based candidates), so no tiebreak was needed there.

**Honest comparison to before adding China** — this is not a one-sided improvement:

| Metric | Before (170 rows) | After (669 rows) | Change |
|---|---|---|---|
| MAE | 267.63 | **130.84** | ↓ 51% (genuinely better) |
| PRED(25) | 38.2% | **20.1%** | ↓ 18.1 points (genuinely worse) |
| `source_dataset` importance | 0.017 | **0.183** | ↑ 10x |

**MAE improved substantially** — expected, given nearly 4x the training data and a tree model that now has real small-project examples to split on instead of guessing. **PRED(25) got clearly worse, and that's stated plainly rather than hidden**: checked why directly on the test set — rows with actual effort under 5 months (17 of 134 test rows, all China-sourced) have a mean *relative* error of 5.13 and only an 11.8% PRED(25) rate, vs. 1.73 mean relative error / 21.4% PRED(25) for the rest. Adding China roughly quadrupled the dataset's size but also massively widened the target's range (0.17 to 2,400 person-months, where before it was 3.6 to 8,211 over a much smaller, more homogeneous pair of datasets) — a relative-error metric like PRED(25) is intrinsically harder to satisfy across a target distribution this heterogeneous, even as the model's absolute predictions (MAE) get meaningfully more accurate. Both things are true at once; this project reports both rather than only the flattering one.

**`source_dataset` importance rising 10x (0.017 → 0.183) is also worth flagging honestly**, not quietly absorbed: the original "Validation the unification actually worked" section above used a *near-zero* `source_dataset` importance as evidence the model wasn't just learning "which dataset did this row come from." That evidence is weaker now — `source_dataset` is the model's 3rd-most-important feature (behind `project_size_kloc` at 0.55 and ahead of `team_experience` at 0.065, with `complexity` at 0.20). The honest read: China's rows are genuinely, systematically different in scale and distribution from COCOMO-NASA/Desharnais (different company, era, and size range), so knowing which collection a row came from legitimately helps the model — this isn't obviously "cheating" the way it would be if `source_dataset` alone could predict effort, but it is a real, worth-noting shift from the original unification's finding, and the paper should present it as such rather than repeating the old near-zero figure unqualified.

### Sanity table — the actual fix, before vs. after adding China

`team_experience=Intermediate, complexity=Nominal`, using the SAME `SOFTWARE_COMPLEXITY_COVERAGE` mechanism `app.py` already computes dynamically from `build_unified_dataframe()` (untouched — no code in `app.py` itself changed for this task):

| KLOC | Before China (months) | After China (months) | Confidence note? |
|---|---|---|---|
| 1 | 25.38 | **5.47** | Yes — 1 KLOC is still just under the real minimum (1.2 KLOC) |
| 5 | 30.64 | **7.69** | No — real coverage now extends below this |
| 20 | 144.25 | 14.32 | No |
| 50 | 423.82 | 44.96 | No |
| 200 | 1,661.02 | 71.20 | No |
| 1,000 | 2,407.61 | 894.58 | No |

**Nominal-complexity coverage, directly**: before China, the smallest real Nominal-complexity project on record was 10 KLOC / 48 person-months (10 total Nominal rows in the whole dataset). After China, `SOFTWARE_COMPLEXITY_COVERAGE['n']` = **75 rows**, smallest real example **1.2 KLOC / 1.4 person-months** — a genuinely small, real project, not an imputed or extrapolated one (China rows do have imputed `complexity`, but the KLOC and effort values themselves are real historical data). 1 KLOC now predicts **5.47 months** — solidly in a "few months for a tiny project" range, down from 25.38 months (the previous fix) and 44.97 months (the original bug) — and the confidence-note mechanism correctly still flags it as marginally below the real minimum (1.2 KLOC) rather than silently presenting it with false confidence.

**The mechanism genuinely didn't need touching, confirmed rather than assumed**: `SOFTWARE_COMPLEXITY_COVERAGE` and `predict_software()`'s confidence-note logic in `app.py` were not modified for this task. Re-running them after retraining shows the coverage numbers above update automatically (read live from `build_unified_dataframe()` at import time), and the note continues to fire correctly for genuinely unsupported inputs: `complexity='vl'` (Very Low) still has **zero** real rows anywhere in the combined 669-row dataset — COCOMO-NASA itself never rated any project "vl", and China's imputed complexity is drawn from COCOMO-NASA's own distribution, so it can never manufacture a `vl` row either — and every `vl` query at every KLOC level still gets the "no real historical project... has this complexity at all" note, exactly as designed before China existed.

**Team Size — still absent, now checked against a third dataset too**: China's only people-related column (`Resource`) was investigated above and found to be an undocumented, weakly/non-monotonically-correlated value — not a headcount, and not treated as `team_experience` directly. The retrained model's feature set is unchanged: `team_experience`, `project_size_kloc`, `complexity`, `source_dataset`. No `team_size` column exists in any of the three source datasets now unified here.

Saved as `models/final_model_software.pkl`, `models/final_scaler_software.pkl`, `models/final_encoder_software.pkl`, `data/processed/software_unified.csv` (669 rows) — same filenames as before, now trained on the 3-dataset union.

## Region-selectable cost rate (business logic, not a model change)

Software's derived "Predicted Cost" ($/person-month × effort) is now region-selectable across 6 markets instead of a single hardcoded constant — a business-rate lookup layered on top of the unaffected effort prediction above, not a retrain or a model change. Full writeup (all 6 rates and sources, the salary-vs-vendor-rate methodology caveat, and why Construction deliberately does not get this) is in `results/cost_regions_notes.md`.
