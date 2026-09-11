# EstimateX — Live Demo Script

Three runs, picked to show the model genuinely responding to input (not returning a fixed number) and to show both domains. All three were run live against the app before writing this script — the numbers below are real output, not projected.

## Setup (do this before walking on stage)

```bash
cd EstimateX
source venv/bin/activate
python app/app.py
```

Open `http://127.0.0.1:5000/` in a browser. Leave the terminal visible on a second screen if possible — an empty, error-free log while you click through is itself part of the proof this is real.

---

## Run 1 — Small, straightforward COCOMO project

**Say before submitting:** "Let's start with a small, fairly standard embedded project."

**Type in:**
| Field | Value |
|---|---|
| Project Domain | Aerospace / Embedded Systems (NASA-style) |
| Project Complexity | Low |
| Required Reliability | Nominal |
| Team Size | 6 |
| Team Experience | Intermediate |
| Estimated Project Size (KLOC) | 15 |

**Result:** Predicted Effort **92.0 months**, Predicted Cost **$920,060**. Top factor: **Project Size (KLOC)**.

**Say while pointing at the chart:** "For a small project like this, size is what the model leans on most — makes sense, there's not much else to differentiate it yet."

---

## Run 2 — Large, high-complexity COCOMO project

**Say before submitting:** "Now let's push every dial up — a much bigger, much stricter project — and watch the number actually move."

**Type in:**
| Field | Value |
|---|---|
| Project Domain | Aerospace / Embedded Systems (NASA-style) |
| Project Complexity | Very High |
| Required Reliability | Very High |
| Team Size | 40 |
| Team Experience | Expert |
| Estimated Project Size (KLOC) | 300 |

**Result:** Predicted Effort **2043.8 months**, Predicted Cost **$20,437,820**. Top factor: **Project Size (KLOC)**, with **Required Reliability** now the clear second factor (18.8%, up from a much smaller share on the first run).

**Say while pointing at the chart:** "Same model, same form, just bigger numbers in — and the estimate jumps by more than 20x, with reliability now pulling real weight in the explanation. That's the model actually reacting, not a static page."

---

## Run 3 — Desharnais (different domain entirely)

**Say before submitting:** "Now let's switch domains completely — a business/information-systems project instead of an embedded one — so we're using the other trained model."

**Type in:**
| Field | Value |
|---|---|
| Project Domain | Business / Information Systems |
| Project Complexity | Nominal *(not used by this domain — see Q&A prep)* |
| Required Reliability | Nominal *(not used by this domain)* |
| Team Size | 8 |
| Team Experience | Senior |
| Estimated Project Size (KLOC) | 25 |

**Result:** Predicted Effort **27.3 months**, Predicted Cost **$272,889**. Top factor: **Adjusted Function Points** (38.0%), not size in KLOC at all.

**Say while pointing at the chart:** "Different domain, different underlying model, and notice the explanation changes shape too — it's not lines of code driving this one, it's function points, which is exactly the right measure for a business system. That's the point of doing this per-dataset instead of forcing one model on everything."

---

## If something looks off live

- Numbers not changing between runs → check you actually changed the Project Domain dropdown or a size/complexity field; Team Size intentionally doesn't move the prediction yet (see `results/testing_notes.md` and `app/app.py`'s docstring) — don't be caught off guard by this if someone changes only Team Size and asks why nothing moved.
- Server error in the terminal → this has been tested end-to-end (`results/testing_notes.md`, Step 11) with 14 scenarios and zero crashes; if something still breaks live, it's worth screenshotting the terminal output for a follow-up fix rather than improvising an explanation.
