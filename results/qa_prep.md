# EstimateX — Q&A Prep

Every number below is from `results/model_comparison.csv` (Step 7), `results/feature_importance_summary.md` (Step 8), `results/final_model_decision.md` (Step 9), or `results/testing_notes.md` (Step 11) — none of this is the original "Expected Outcomes" forecast.

---

### "Why did you choose Random Forest, and not the others?"

It won the combined MAE + PRED(25) comparison on **both** datasets, not just one metric in isolation:

- **COCOMO-NASA**: Random Forest wins outright — best MAE (219.8), best RMSE (440.3), best MMRE (0.770), best PRED(25) (42.1%). No other model beats it on anything.
- **Desharnais**: Linear Regression has a marginally lower raw MAE (2402.2 vs. 2434.1 — about 1.3% lower), but Random Forest gets roughly **4x as many predictions within 25% of the actual value** (PRED(25) 25.0% vs. 6.3%). We treated that as a near-tie on average error decided by which model is actually more consistent, and picked Random Forest.
- It's also one of only two models (with XGBoost) that support feature-level explainability, which was a hard requirement, not a nice-to-have. Linear Regression's coefficients aren't usable the same way after one-hot encoding, and it was separately flagged as badly unstable in cross-validation on COCOMO-NASA (see below).

Full reasoning: `results/final_model_decision.md`.

---

### "How accurate is this really — what do MAE / RMSE / PRED(25) actually mean?"

- **MAE** (Mean Absolute Error): the average size of the miss, in the target's own units. COCOMO-NASA's Random Forest has MAE 219.8 — predictions are off by about 220 person-months on average.
- **RMSE**: like MAE but penalizes big misses harder. It's notably higher than MAE here (440.3 vs. 219.8), meaning a handful of predictions are much further off than the "typical" case — not every miss is small.
- **MMRE**: average *relative* error. 0.770 on COCOMO-NASA means predictions differ from the true value by about 77% of it, on average — a proportional, not absolute, measure.
- **PRED(25)**: the percentage of test projects where the prediction landed within 25% of the actual value. 42.1% on COCOMO-NASA, 25.0% on Desharnais.

**Honest framing**: this is a real, measured improvement over the plain Linear Regression baseline and over pure guesswork, but PRED(25) here is below the roughly 75% mark that effort-estimation papers often treat as the bar for a "genuinely reliable" estimator. This should be presented as a useful, data-grounded starting point for a conversation about scope and budget — not a number precise enough to commit a fixed-price contract to.

---

### "Why not use deep learning?"

This is already answered in Section II of the paper and we should say the same thing, not invent a new answer: the 2025 literature review we cite found deep learning scoring higher on average than classical ML (85–90% vs. 75–80%), **but** that gain "does not come free" — it needs substantially more data and costs a lot of the interpretability that makes a model useful to a non-expert. Our combined training data is under 200 records (93 COCOMO-NASA + 81 Desharnais); that's nowhere near enough to train a deep network without it just memorizing the training set. Classical ML — specifically a tree-based model — handles small-sample estimation reasonably well *and* stays explainable, which was a founding requirement of this whole project, not an afterthought.

---

### "What happens if I put in a project completely unlike anything in your training data?"

We tested exactly this in Step 11: COCOMO-NASA at 5,000 KLOC against a training maximum of 980, and an equivalent stress test on Desharnais. Random Forest doesn't extrapolate the way a linear formula would blow up — its prediction is bounded by the patterns it actually learned (it effectively falls back to whatever large-project pattern is closest to what it's seen), so you won't get an absurd, runaway number. **But a bounded prediction isn't the same as an accurate one** that far outside the data. This is stated explicitly as a limitation in `results/testing_notes.md` and in the paper's Limitations section: the further a project sits from the training distribution, the more that caveat should weigh on how the number is used.

---

### "How is this different from just using COCOMO?"

COCOMO is a fixed formula calibrated on a specific historical set of projects. The moment the team, tooling, or project type changes from what it was calibrated on, the formula stops adapting — it can't learn anything new. EstimateX instead learns the relationship between project attributes and effort directly from historical data, can in principle be retrained as more data comes in, and — critically — works across **two independently collected domains** (NASA-style embedded systems and Desharnais-style business software) with the domain handled as just an input choice rather than needing a whole separate hardcoded model or formula per domain. It also explains each individual prediction instead of only handing back a number.

---

### "What's the explainability output actually showing — is it trustworthy?"

It's Random Forest's built-in feature importance, further weighted **per request** by how unusual that specific input's values are relative to a typical project (importance × deviation from typical). That's a documented, deliberate simplification (see `app/app.py`'s docstring) — it is **not** SHAP or a formally rigorous per-instance attribution method, and we should say that plainly if asked directly.

Is it trustworthy? It reflects exactly what the trained model actually relies on — we cross-checked it against domain intuition in Step 8 (e.g. project size dominates both datasets, which matches "bigger projects cost more" and is a real, defensible finding), so it's not made up or decorative. But it's an approximation of "why this number," grounded in the model's own structure, not a causal explanation of software engineering reality.

---

### "What would you do next if you had more time?"

- More, and more recent, data across more domains — the paper's own stated future work. Both datasets here are decades old and predate cloud-native and agile practices.
- Real per-instance explainability (SHAP) instead of the current importance × deviation heuristic.
- Hyperparameter tuning — deliberately out of scope for Step 6's first pass; PRED(25) has real room to improve.
- A genuine feature for Team Size once a dataset that actually measures headcount is available (see limitation below).
- Production-grade serving (a real WSGI server, not Flask's dev server) if this ever needed to handle real traffic.

---

## Limitations a sharp question might target (from Step 11 testing)

- **Team Size doesn't affect either prediction right now.** We tested this directly — identical COCOMO-NASA input with `team_size=2` vs. `team_size=200` gives the exact same prediction (290.2 months, both times). Neither raw dataset has an actual headcount column (COCOMO's drivers are capability *ratings*, Desharnais's are experience in *years*), so the field is collected and shown in the UI honestly-labeled as "informational for now," not silently faked. **Be ready for this if someone tests it live and asks why nothing changed.**
- **Project Complexity and Required Reliability only affect the COCOMO-NASA path.** Desharnais has no analogous columns, so those two fields are accepted but ignored when that domain is selected.
- **"Predicted Cost" is illustrative**, not something either model was trained on — neither dataset has a currency column. It's effort converted to a dollar figure using an assumed $10,000/person-month constant, clearly labeled in the UI and adjustable in `app.py`.
- **Desharnais's KLOC input is a named heuristic conversion** (Capers Jones' ~100 lines-of-code-per-function-point "backfire" rule), not a measured relationship, since Desharnais has no lines-of-code column at all.
- **Combined training data is under 200 records.** This limits how complex a model can be trained before it starts memorizing instead of learning, and is the main reason hyperparameter tuning and deep learning were both out of scope.
- **Linear Regression was badly unstable on COCOMO-NASA** (cross-validation R² of **-7.11**, found in Step 6) — a real, honestly-reported finding, not glossed over. Cause: 48 one-hot-encoded features fit on ~59 training rows per fold, without regularization, is a numerically ill-posed problem for plain linear regression. This is additional evidence for why Random Forest, not Linear Regression, was the right choice despite its narrow MAE win on Desharnais.
- **COCOMO-NASA's PRED(25) (42.1%) is notably higher than Desharnais's (25.0%)** even though the same algorithm won both. Plausible reasons if asked: COCOMO-NASA has richer engineered features (15 ordinal effort-driver ratings) than Desharnais, and Desharnais's test set is only 16 rows, so each prediction is worth 6.25 percentage points of PRED(25) — a coarser, more volatile measure than COCOMO-NASA's 19-row test set.
