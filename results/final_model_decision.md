# EstimateX — Final Model Decision (Step 9)

**Chosen algorithm: Random Forest**, deployed for both COCOMO-NASA and Desharnais.

## Selection criteria

Per dataset, the winning model was chosen using MAE (lower is better) and PRED(25) (higher is better) TOGETHER, not either metric in isolation. When the two metrics disagreed, a model was preferred on PRED(25) only if its MAE was within 5% of the best MAE in that dataset (a near-tie on average error, decided in favor of the model that lands within 25% of the true value more often) — otherwise the MAE leader was kept. This tolerance is a stated methodology choice, not an arbitrary tiebreak.

## COCOMO-NASA

- Best MAE: **Random Forest** (219.83)
- Best PRED(25): **Random Forest** (42.1%)
- **Winner: Random Forest.** Random Forest has both the lowest MAE (219.83) and the highest PRED(25) (42.1%) — wins outright, no tiebreak needed.

## Desharnais

- Best MAE: **Linear Regression** (2402.17)
- Best PRED(25): **Random Forest** (25.0%)
- **Winner: Random Forest.** Linear Regression has the lowest MAE (2402.17) and Random Forest has the highest PRED(25) (25.0%). Random Forest's MAE (2434.10) is only 1.33% higher than Linear Regression's — within the 5% tolerance — while its PRED(25) is 18.8 percentage points higher. Preferring PRED(25) here: a near-tie on average error decided in favor of the model that lands within 25% of the actual value far more often, which matters more in practice than a small MAE difference.

## Final decision

Same algorithm (Random Forest) wins the combined MAE+PRED(25) comparison on BOTH datasets — deploying it directly, no fallback rule needed.

This also happens to align with Step 8's explainability findings: Random Forest is one of the two tree-based models with feature-importance support, so choosing it keeps EstimateX's "not a black box" explainability intact rather than trading it away for a small accuracy difference.

## Tradeoffs acknowledged

- **COCOMO-NASA — RMSE**: SVR has the best RMSE (353.27) on COCOMO-NASA, beating Random Forest's RMSE (440.32). SVR was not chosen: its PRED(25) (10.5%) is far worse than Random Forest's (42.1%), and it has no feature-importance support at all (Step 8's scoping note), so it fails the explainability requirement regardless of its RMSE edge.
- **Desharnais — MAE and RMSE**: Linear Regression narrowly has the lowest raw MAE (2402.17 vs Random Forest's 2434.10 — a 1.33% difference) and the lowest RMSE (3075.30 vs 3343.40) on Desharnais. It was not chosen: the MAE/RMSE gap is small enough to be within noise on a 16-row test set, while its PRED(25) (6.3%) is four times worse than Random Forest's (25.0%), and — as already flagged in Steps 6-8 — Linear Regression's coefficients aren't usable for feature-importance-style explainability the way Random Forest's are, and its COCOMO-NASA CV performance (R² mean -7.11) was separately flagged as badly unstable, reinforcing that it isn't the more reliable choice across this project even where it narrowly leads on one metric.
- **Neither dataset's PRED(25) is high in absolute terms** (42.1% and 25.0% respectively, both well under the conventional 75% "good estimator" benchmark used in effort-estimation literature) — Random Forest is the best available choice among the four models tried, not a model that has solved effort estimation on these datasets outright. Worth stating plainly in the paper rather than overselling the result.

## Deployed artifacts

| File | Contents |
|---|---|
| `models/final_model_cocomo.pkl` | Fitted Random Forest model, COCOMO-NASA |
| `models/final_model_desharnais.pkl` | Fitted Random Forest model, Desharnais |
| `models/final_scaler_cocomo.pkl` | Fitted `StandardScaler` for COCOMO-NASA's numeric + ordinal-encoded columns |
| `models/final_encoder_cocomo.pkl` | Dict `{"ordinal": OrdinalEncoder, "onehot": OneHotEncoder, "binary_forg": LabelEncoder}` — COCOMO-NASA needs all three to reproduce its exact feature space |
| `models/final_scaler_desharnais.pkl` | Fitted `StandardScaler` for Desharnais's numeric columns |
| `models/final_encoder_desharnais.pkl` | Fitted `OneHotEncoder` for Desharnais's `Language` column |

All of the above are re-saves (joblib load -> joblib dump under a new name) of the exact artifacts Steps 4 and 6 already fit — nothing was refit, so these are guaranteed to match the transform used to produce the Step 7/8 numbers.
