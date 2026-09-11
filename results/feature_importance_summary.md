# EstimateX — Feature Importance & Explainability Summary

This summarizes what Random Forest and XGBoost — the two tree-based models trained in Step 6 — say drives their effort/cost predictions on each dataset. It's written for direct use in the paper's explainability section and for talking through in the presentation.

## COCOMO-NASA: what matters most

Both models agree on the same two leading factors, even though they weigh them differently:

- **`equivphyskloc`** (project size, in thousands of lines of code) and **`time`** (the COCOMO rating for how tight the runtime/execution-time constraint is) are the top two drivers in both models.
- **Random Forest** spreads its attention more evenly: size (`equivphyskloc`, ~39%) edges out the time constraint (`time`, ~25%), with reliability requirement (`rely`, ~10%) and project `year` (~10%) as clear secondary factors.
- **XGBoost** concentrates much more heavily on a single factor: the `time` constraint alone accounts for ~70% of its importance, with size (`equivphyskloc`, ~8%), analyst capability (`acap`, ~6%), and the development-facility flag (`forg`, ~6%) trailing well behind.
- Everything else — the specific NASA center, project category (`cat2`), development mode, and most of the other COCOMO effort-driver ratings — contributes only marginally (each under 3%) in both models.

**In plain language:** for NASA-style projects, *how big the system is* and *how strict the runtime performance requirement is* are what really move the effort estimate. The other 15 effort-driver ratings COCOMO asks for (team capability, tool use, language experience, etc.) matter, but only at the margins — at least for these 93 historical projects.

## Desharnais: what matters most

- **`PointsAdjust`** (adjusted function points — a standard measure of overall system size) is the single most important feature in both models by a wide margin (~32% in Random Forest, ~40% in XGBoost).
- **`Length`** (project duration) and **`PointsNonAjust`** (unadjusted function points, another size measure) are consistently in the top few for both models as well.
- **Random Forest** additionally weighs `Transactions` (~12%) and `Entities` (~8%) — both size-related function-point sub-counts — quite highly, painting a fairly consistent "size drives effort" picture.
- **XGBoost** stands out by giving one specific programming-language category (`Language_3`, ~23%) unusually high importance — much more than Random Forest does (~3%) — while giving less weight to `Transactions` and `Entities` than Random Forest does. This kind of language-specific effect is plausible (different languages genuinely have different productivity profiles) but with only 77 training rows split across 3 language categories, it's also exactly the kind of thing that could be an artifact of a small sample — worth treating as a hypothesis to check on more data, not a settled conclusion.
- **`TeamExp`** and **`ManagerExp`** — team/manager experience — rank *surprisingly low* in both models (each under 2% importance). This runs against the common intuition that a more experienced team should visibly reduce effort, and is worth calling out explicitly as a counterintuitive finding rather than glossing over it — it may reflect the small sample size, or that experience's effect is real but non-linear in a way these models' default settings aren't capturing well.

## Does the same thing matter across both datasets?

**Yes, at the level of the underlying concept: project size is the dominant driver of effort/cost in both datasets and in both models.** COCOMO-NASA's `equivphyskloc` (lines of code) and Desharnais's `PointsAdjust`/`Length`/`PointsNonAjust`/`Transactions` (function-point-based size measures) are different ways of measuring the same underlying idea — "how big is this system" — and in both datasets, size-related features are what the models lean on most.

Beyond that top-level agreement, the two datasets diverge, which makes sense given they describe different kinds of projects: COCOMO-NASA's second-biggest factor is a *technical constraint* (the `time`/runtime-performance rating), reflecting its embedded/real-time NASA/avionics project mix, while Desharnais's secondary factors are all still *other size measures* (`Length`, `Transactions`, `Entities`), reflecting its business/transaction-processing project mix. Neither dataset shows team or individual experience ratings as a leading factor, despite those being intuitively appealing explanations — a point worth being upfront about rather than overselling the models' "explanations."

## Scoping note: why only Random Forest and XGBoost

Linear Regression and SVR were **not** analyzed in this step, and won't be, for reasons intrinsic to how each model works — this is a deliberate scoping decision, not something overlooked:

- **Linear Regression** does technically have coefficients, but they aren't directly comparable to `.feature_importances_` or to each other here: after one-hot encoding, a plain coefficient's magnitude is entangled with how many categories a dummy variable set has and with correlated/collinear predictors (Step 6 already found COCOMO's Linear Regression to be numerically unstable — CV R² of -7.1 — for exactly this reason). Treating those coefficients as "importance" would be misleading.
- **SVR**, with the default RBF kernel used in this project, doesn't expose *any* coefficient or importance concept — it isn't a linear model in the original feature space, so there's nothing analogous to extract.

As a result, **explainability in EstimateX is currently tied to whichever tree-based model (Random Forest or XGBoost) Step 9 selects as the best-performing model** for each dataset. If Step 9 ultimately favors Linear Regression or SVR on accuracy grounds for a given dataset, that would come at the cost of losing this kind of feature-level explanation — a trade-off worth naming explicitly when Step 9's model choice is justified in the paper.
