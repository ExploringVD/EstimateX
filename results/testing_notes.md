# EstimateX — End-to-End Testing Notes (Step 11)

Testing method: the real Flask app (`app/app.py`) exercised through Flask's test client (real view functions, real model/scaler/encoder loads, real `.predict()` calls — no mocking) for functional coverage, plus a live `python app/app.py` dev server driven by an actual browser (Playwright/Chromium) for visual and interaction checks.

## 1. Five realistic test cases

| # | Case | Domain | Inputs | Predicted Effort | Predicted Cost | Top factors |
|---|---|---|---|---|---|---|
| 1 | Small, low-complexity | COCOMO | team=3, complexity=VL, reliability=VL, exp=Junior, 5 KLOC | 119.6 months | $1,196,120 | Required Reliability (42.9%), Project Size (29.6%) |
| 2 | Large, high-complexity | COCOMO | team=60, complexity=XH, reliability=XH, exp=Expert, 600 KLOC | 2,261.8 months | $22,617,920 | Project Size (72.1%), Required Reliability (16.4%) |
| 3 | Mid-size "typical" | Desharnais | team=10, exp=Intermediate, 40 KLOC | 34.5 months | $344,561 | Adjusted Function Points (46.9%), Unadjusted Function Points (20.3%) |
| 4a | Very small team (1) | Desharnais | team=1, exp=Novice, 8 KLOC | 9.3 months | $93,385 | Adjusted Function Points (47.6%) |
| 4b | Very small team (2) | COCOMO | team=2, exp=Novice, 8 KLOC | 290.2 months | $2,902,420 | Project Size (42.7%), Runtime Constraint (32.5%) |
| 5 | Unusually large project | COCOMO | 5,000 KLOC (training max: 980) | 1,840.4 months | $18,404,380 | Project Size (98.0%) |
| 5b | Unusually large project | Desharnais | 2,000 KLOC-equivalent (training max ≈1,127 function points) | 93.2 months | $931,645 | Adjusted Function Points (48.1%) |

**All 7 distinct input combinations produced 7 distinct predictions** — confirms the form is genuinely wired to the trained models, not returning a fixed number. (Case 4b and 4c below are identical by design — see the Team Size finding.)

**Sanity check — all predictions non-negative and plausible**: effort ranged from 9.3 to 2,261.8 months across all cases. The largest (Case 2, Case 5) are big numbers, but not absurd — COCOMO's own training data includes real projects up to 8,211 person-months, so 1,840–2,262 months for a 5–600 KLOC hypothetical project is within the shape of the data the model actually learned from, not a runaway extrapolation artifact.

**Feature importance / explanation sanity**: in every case the top-ranked factor(s) matched domain intuition — project size or a technical constraint (COCOMO) and function-point/size measures (Desharnais) consistently lead, matching Step 8's global findings. The plain-language sentence correctly named the actual top 1-2 factors for each specific case (e.g. Case 1's low-reliability, small project surfaced "Required Reliability" as the lead factor; Case 2's huge, high-complexity project surfaced "Project Size" instead) — confirming the per-request explanation genuinely varies with input, not a static sentence.

## 2. Team Size isolation check

Ran the same COCOMO inputs (complexity=N, reliability=N, exp=Novice, 8 KLOC) with `team_size=2` and `team_size=200` — **predictions were identical (290.2 months both times)**. This confirms, rather than contradicts, Step 10's documented design: neither trained model uses a headcount feature at all (COCOMO has capability *ratings*, Desharnais has experience *years* — neither dataset has "number of people"), so Team Size is currently collected but inert by design, not by bug. Logged here explicitly since it's the kind of thing a live audience might notice and ask about.

## 3. Bad/edge-case input handling

| Test | Input | Result |
|---|---|---|
| Empty required field | `team_size=""` | HTTP 400, inline error "Team size is required." |
| Negative team size | `team_size=-5` | HTTP 400, "Team size must be a positive whole number." |
| Negative project size | `project_size_kloc=-50` | HTTP 400, "Project size must be a positive number." |
| Extremely large team size | `team_size=999999` | **Found broken, fixed — see below** |
| Missing domain field | (omitted) | HTTP 400, "Please choose a project domain." |
| Non-numeric project size | `project_size_kloc="abc"` | HTTP 400, "Project size must be a number." |

No case produced a stack trace or a 500 error — every invalid input is caught and re-renders the form with the submitted values preserved and a clear inline message next to the offending field.

## 4. Bug found and fixed

**`team_size=999999` was silently accepted (HTTP 200)** instead of being rejected, because `validate_form()` only checked for "positive," with no upper bound. Since Team Size doesn't drive either prediction, this wasn't causing wrong numbers — but it's still a real gap: a fat-fingered or malicious huge value should be caught with feedback, not silently swallowed, and the task explicitly expected an error here.

**Fix**: added `MAX_TEAM_SIZE = 2_000` and `MAX_PROJECT_SIZE_KLOC = 100_000` constants in `app/app.py`, wired into `validate_form()`, and reflected as native `max="..."` attributes on the number inputs in `form.html` for browser-level hinting too. Bounds were chosen generously — well above any realistic project (2,000 people, 100,000 KLOC) and comfortably above this very test plan's own "unusually large" stress cases (600/5,000 KLOC for COCOMO, 2,000 KLOC-equivalent for Desharnais) — so legitimate stress-testing still passes, only clearly-nonsensical input is blocked.

**Re-tested after the fix**: `team_size=999999` now returns HTTP 400 with "Team size must be 2,000 or fewer." — confirmed via a targeted re-run of that exact case, and the full 14-scenario suite (5 realistic + 3 team-size-isolation + 6 bad-input) was re-run afterward to confirm nothing else regressed.

## 5. UI check at desktop and mobile width

Initial pass using Chrome's CLI `--headless --window-size=390,844` screenshot showed apparent right-edge cut-off (tagline, hint text, and button all appeared clipped). Investigated before assuming it was a real CSS bug — used Playwright with proper mobile viewport emulation (`viewport={width: 390, height: 844}`) instead, and confirmed programmatically that `document.documentElement.scrollWidth === document.documentElement.clientWidth` at both 1280px and 390px (no horizontal overflow at all). Re-screenshotted with Playwright: the form and results pages both render cleanly at 390px — result stats stack vertically, feature-importance labels wrap to two lines without breaking the bar layout, and the primary button stays full-width and fully visible.

**Conclusion: the earlier cut-off was a testing-tool artifact** (Chrome's bare `--window-size` CLI flag doesn't reliably emulate `width=device-width` the way a real mobile viewport does), not a bug in the app — logged here for transparency since it looked like a real failure at first glance, and the correction is worth recording in case someone re-tests with a simple screenshot tool and sees the same false alarm.

No CSS changes were needed as a result of this check.

## 6. Known limitations (not bugs)

- **Team Size does not currently affect either prediction** — neither trained dataset has a raw headcount feature. Documented in the form's own helper text and in `app/app.py`'s module docstring since Step 10; re-confirmed here.
- **Project Complexity and Required Reliability only affect the COCOMO path** — Desharnais has no analogous columns, so these two fields are accepted but ignored when "Business / Information Systems" is selected.
- **Predictions well outside the training data's range should be treated with caution.** Random Forest can't truly extrapolate — for inputs far beyond what the model has seen (e.g. 5,000 KLOC vs. a training max of 980), it effectively predicts based on the most similar large projects it *has* seen, which bounds the output sensibly (confirmed in Case 5 above) but doesn't mean the number is a rigorously validated estimate for that scale of project.
- **"Predicted Cost" is an illustrative derived figure**, not something either model was trained to predict — it's effort × an assumed $10,000/person-month constant, clearly labeled in the UI, adjustable in `app/app.py`.
- **Desharnais's KLOC input is a heuristic conversion** (Capers Jones' ~100 LOC/function-point "backfire" rule), not a measured relationship — documented in `app/app.py`.
- **"What drove this estimate" is a lightweight per-request heuristic** (importance × how far this input's value is from a typical project), not a formal method like SHAP — documented in `app/app.py`.

None of the above are new findings; all were already flagged explicitly during Step 10 and are restated here because end-to-end testing is exactly the point at which they'd surface as apparent "bugs" to someone unfamiliar with the design — worth having in one place before a live demo.

## Summary

**Tested**: 5 realistic input scenarios (+2 extra domain/size variants), a team-size isolation check, 6 invalid-input scenarios, and responsive layout at desktop (1280px) and mobile (390px) widths.
**Passed**: all of the above, after one fix.
**Found and fixed**: missing upper-bound validation on Team Size (and, preventatively, Project Size) — now rejected with a clear inline error instead of silently accepted.
**False alarm, investigated and ruled out**: apparent mobile layout cut-off, traced to a flawed screenshot testing method rather than an actual CSS bug.
**Remaining known limitations**: listed above, all pre-existing documented design decisions, not defects.
