# Model Card: GamePulse Churn Classifier

## Summary

GamePulse estimates the probability that a fictional player will become inactive during the 30 days after the observation cutoff. The model supports a portfolio dashboard focused on gaming analytics and retention planning.

## Model and pipeline

- Algorithm: Random Forest classifier
- Preprocessing: median imputation for numeric inputs; most-frequent imputation and one-hot encoding for categorical inputs
- Training data: 2,250 synthetic records (75%)
- Holdout data: 750 synthetic records (25%)
- Split: stratified with random seed 42
- Default classification threshold: 0.50
- Risk bands: Low `< 0.30`, Medium `0.30–0.59`, High `≥ 0.60`

## Fixed holdout performance

| Metric | Value |
|---|---:|
| ROC AUC | 0.755 |
| Accuracy | 0.716 |
| Precision | 0.557 |
| Recall | 0.619 |
| F1 | 0.586 |

The interactive dashboard allows the decision threshold to be changed so the trade-off between precision and recall is visible.

## Intended use

- Demonstrate an end-to-end machine-learning workflow
- Explore synthetic player engagement and churn patterns
- Practice prioritizing retention reviews with model-assisted analytics
- Discuss model evaluation, thresholds, and responsible use in interviews

## Out-of-scope use

- Production decisions about real players
- Individual targeting without consent and human review
- Safety-critical, financial, employment, eligibility, or disciplinary decisions
- Claims about a real game's retention or commercial performance

## Limitations

- All relationships are generated and are simpler than real game behavior.
- The dataset has no temporal drift, seasonality, content releases, or experiments.
- Holdout results measure fit to the synthetic generator, not generalization to a real game.
- Retention actions are illustrative and have not been evaluated in controlled experiments.
- Group-level fairness is not established. Region is available for descriptive filtering but is excluded from the model feature set.

## Monitoring required before real use

A production adaptation would require consent and privacy review, time-based validation, calibration checks, drift monitoring, subgroup performance analysis, experiment-based measurement of retention actions, and a documented human-review process.

## Reproducibility

The dataset generator, split, classifier, and tests use fixed seeds. Regenerate and verify with:

```bash
python scripts/generate_data.py
python -m unittest discover -s tests -v
```
