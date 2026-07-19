# GamePulse: Player Churn & Engagement Analytics

GamePulse is an end-to-end gaming analytics portfolio project that predicts 30-day player churn, explores engagement patterns, and turns model output into practical retention actions.

The project uses **3,000 deterministic synthetic player records**. It contains no real players, personal data, or proprietary game telemetry.

## What the dashboard answers

- Which players have the highest predicted churn risk?
- How do engagement and churn differ by platform and preferred game mode?
- Which high-value players should a retention team review first?
- What action is recommended for each retention segment?
- How does the model perform when the classification threshold changes?

## Features

- Interactive Streamlit dashboard with platform, mode, region, and risk filters
- Random Forest churn model built in a reproducible scikit-learn pipeline
- Holdout evaluation with ROC AUC, precision, recall, F1, ROC curve, and confusion matrix
- Action-oriented segments such as `High-value at risk` and `New player friction`
- Downloadable player-review and retention-action queues
- Automated data, model, analytics, and dashboard startup tests
- GitHub Actions workflow for repeatable checks

## Model result

The fixed 3,000-player dataset is split into a stratified 75% training set and 25% holdout set.

| Metric | Holdout result |
|---|---:|
| ROC AUC | 0.755 |
| Accuracy | 71.6% |
| Precision | 55.7% |
| Recall | 61.9% |
| F1 score | 0.586 |

These results describe only the included synthetic dataset. They are not evidence of production performance on a real game.

## Project structure

```text
.
├── app.py                         # Streamlit dashboard
├── data/gamepulse_players.csv     # Deterministic synthetic dataset
├── scripts/generate_data.py       # Dataset generator
├── src/
│   ├── analytics.py               # Segments, actions, and filters
│   ├── data.py                    # Synthetic telemetry and validation
│   └── modeling.py                # Training, scoring, and evaluation
├── tests/                         # Automated test suite
├── DATA_DICTIONARY.md
├── MODEL_CARD.md
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the data and tests

```bash
python scripts/generate_data.py
python -m unittest discover -s tests -v
```

## Responsible use

GamePulse is an educational portfolio demonstration. A churn score should never be used to target or penalize real people without validation, monitoring, privacy controls, and human review. See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations, and evaluation details.

## Author

**Yashraj Adsul**<br>
AI & Data Science Engineering Student<br>
[LinkedIn](https://www.linkedin.com/in/yashraj-adsul) · [GitHub](https://github.com/yashrajadsul165)

## License

This project is available under the [MIT License](LICENSE).
