# EstimateX — Region-Selectable Cost Rate Notes

Cross-referenced from `app/app.py`'s module docstring (assumption 10). Covers: why this is software-only, all 6 rates and their sources, the methodology caveat, and the sanity checks run before calling it done.

## This is a business-rate lookup, not a machine-learning change

Software's "Predicted Cost" has always been a **derived** figure: predicted effort (person-months) × an assumed $/person-month rate. None of the three training datasets (COCOMO-NASA, Desharnais, China — see `results/software_unification_notes.md`) contain any real cost or salary data; the trained model only ever learned **effort**. The $/month multiplier was always an external business assumption applied after the model runs, not something the model itself produces. That's exactly why making it region-selectable is safe: `predict_software()`'s effort prediction is completely unaffected by which region is chosen — only the multiplier applied to that same effort number afterward changes. Verified directly (see "Sanity checks" below): submitting the identical project with `region=us` vs. `region=india` produces the **exact same** predicted effort (5.5 months in both cases) and a cost ratio that matches the rate ratio exactly.

## Why Construction does NOT get a region selector

Construction's cost is different in kind, not just in unit. It is the **actual regression target** the construction model was trained on, straight from the UCI Residential Building dataset, in that dataset's native unit (×10,000 Iranian Rial) — not a derived effort×rate estimate. That model already has its own genuine, data-grounded locality signal: the Locality Zone (1-20) input feature is real Tehran-area zone codes from the source data, not a placeholder. Adding a "United States / India / Vietnam..." region selector on top of Construction would mean multiplying an Iranian-Rial-denominated, already-locality-aware ML prediction by a generic software-developer monthly salary figure — mixing two unrelated units (IRR vs. USD) and two unrelated economies (residential construction in Tehran-area zones vs. global software-developer labor markets) with no honest basis for the combination. So Construction's cost display, its form fields, `predict_construction()`, and its results.html branch are all left exactly as they were — no region field, no code changes there at all.

## The 6 regions, rates, and sources

Sourced and spot-checked 2026-09-12.

| Region | Rate ($/mo) | Basis | Source |
|---|---|---|---|
| United States | 9,300 | Raw average salary | ZipRecruiter, Sept 2026 US national average software developer salary: $111,845/yr ÷ 12 = $9,320/mo (rounded to $9,300) |
| United Kingdom | 6,000 | Raw average salary | Payprecision 2026: GBP56,914/yr avg ≈ GBP4,743/mo, converted at a static approximate GBP1=$1.27 |
| Western Europe (Germany) | 5,800 | Raw average salary | jobvector 2026 (via WBS Coding School): EUR64,700/yr avg ≈ EUR5,392/mo, converted at a static approximate EUR1=$1.08 |
| Eastern Europe (Poland) | 3,600 | Raw average salary | EuroTopTech 2026: EUR35-45K/yr national median, midpoint EUR40K/yr ≈ EUR3,333/mo, same static conversion |
| India | 4,500 | Blended offshore/vendor rate | Acquaintsoft 2026 rate card, midpoint of $3,500-5,500/mo |
| Southeast Asia (Vietnam) | 4,000 | Blended offshore/vendor rate | Linnoedge 2026, midpoint of $3,440-4,900/mo |

All conversion rates (GBP/EUR → USD) are **static, approximate, and labeled as such** — same spirit as `INR_PER_USD` elsewhere in this app: a fixed, documented demo rate, not a live fetch.

### Spot-checks actually performed (not trusted blindly)

- **United States**: confirmed directly against ziprecruiter.com's live "Software Developer Salary" page. The $111,845/yr figure is real and current as of the check date. $111,845 / 12 = $9,320.42/mo — the prompt's $9,300 is a reasonable round-number approximation of this, off by about 0.2%. Used $9,300 in code as specified, noting the more precise figure here for transparency.
- **India**: confirmed against acquaintsoft.com's 2026 rate card. Real offshore hourly rates there run $20-40/hr generally, or $28-38/hr specifically for a vetted mid-level hire with contractual protections. At roughly 160 hours/month, that converts to approximately $3,200-6,400/mo (general range) or $4,480-6,080/mo (mid-level specific) — the same order of magnitude as the claimed $3,500-5,500/mo midpoint, though the exact midpoint isn't independently reproducible to the dollar from a single hourly figure; it's directionally consistent, not a bit-for-bit match.
- UK, Germany, Poland, and Vietnam figures were not independently re-verified against their primary sources this round — flagged here rather than silently presented with the same confidence as the two spot-checked ones.

### Methodology caveat — stated plainly, not smoothed over

**US/UK/Germany/Poland are raw average salaries. India/Vietnam are blended offshore-vendor rates.** This is not one uniform methodology end to end — it's what's actually well-documented for each type of market (raw salary surveys are the norm for the first group; vendor rate cards are the norm for the offshore-outsourcing markets in the second group). This is also *why* India ($4,500) and Poland ($3,600) land close together despite the usual "Eastern Europe generally costs more than India" framing found in casual outsourcing comparisons — a vendor rate (India) and a raw salary average (Poland) aren't measuring quite the same thing, so their proximity here is partly a methodology artifact, not a precise economic ranking.

**Adjacent regions in this table should not be read as precisely ranked.** The real, robust signal is the **~2.6x spread** between the cheapest market (Poland, $3,600/mo) and the most expensive (United States, $9,300/mo) — that order-of-magnitude difference is meaningful and worth showing users; the fine-grained ordering of, say, India vs. Poland vs. Southeast Asia is not something this table should be read as asserting with precision.

## Implementation

- `app/app.py`: `COST_REGIONS` (list of `(value, label, rate)`) replaces the old single `COST_PER_PERSON_MONTH = 10_000` constant. `COST_REGION_RATES`/`COST_REGION_LABELS` are lookup dicts derived from it; `DEFAULT_COST_REGION = "us"` preserves prior behavior for anyone who doesn't touch the new field. `predict_software()` now takes `cost_region`, looks up the rate, and computes `predicted_cost = predicted_effort_months * cost_rate` — the only change to that function's cost math; its effort-prediction code path is untouched.
- `app/templates/form.html`: a "Cost Basis Region" dropdown added inside the existing `#software-fields` toggle group (same group Team Experience / Project Size / Complexity / Planned Start Month/Year already live in) — so it's shown for all three software domain values and hidden for Construction automatically, via the exact same domain-toggle JS (`app/static/app.js`) already driving every other software-only field. No JS changes were needed; `app.js` operates generically over every `input, select` inside whichever group is toggled.
- `app/templates/results.html`: the software cost caption now reads "Cost assumes $X/person-month ({region} reference rate, 2026) — pick a different region on the form for a different market's typical rate, converted at ₹85/$1 (static demo rate)" — `cost_rate` and `cost_region_label` come from `predict_software()`'s return value, resolved per-request. Construction's caption and results branch are byte-for-byte unchanged.
- `validate_form()`: `cost_region` is validated for software domains the same way `complexity`/`team_experience` already are (must be one of `COST_REGION_VALUES`), not validated at all for Construction.

## Sanity checks performed

1. **1 KLOC / Nominal complexity / Intermediate experience, region=India vs. region=United States**: both requests predicted the **identical** 5.5 months of effort (confirming region has zero effect on the ML prediction). Cost came back as $50,909 (US) and $24,633 (India). Ratio: 24,633 / 50,909 = 0.4839; the rate ratio 4,500 / 9,300 = 0.4839 — an exact match, confirming cost scales precisely proportionally to the chosen region's rate and nothing else.
2. **One Construction request** (locality=5, floor_area_m2=850, lot_area_m2=300, prelim costs, planned_duration=18, unit_price_start=9.5) was submitted end to end: predicted cost came back as 204 (×10,000 IRR) — identical to pre-existing test runs from before this change, confirming no regression. The rendered results page contains zero occurrences of "cost_region," "Cost Basis Region," or any region label — the feature is completely invisible to the Construction path, not just visually hidden.
3. Playwright end-to-end pass on both domains: zero console errors; `#cost_region` is visible on the software form and confirmed `is_visible() == False` after switching the domain dropdown to Construction, using the same toggle mechanism as every other domain-conditional field.
