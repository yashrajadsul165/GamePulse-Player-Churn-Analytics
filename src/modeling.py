"""Model training, scoring, and evaluation utilities for GamePulse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    roc_points: pd.DataFrame
    test_probabilities: np.ndarray
    test_labels: np.ndarray


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create a reproducible preprocessing and classification pipeline."""

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=240,
        max_depth=9,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=random_state,
        # A single worker is more reliable on small free-tier deployment
        # containers, and this dataset is intentionally compact.
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocessing), ("classifier", classifier)])


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Calculate classification metrics at a chosen probability threshold."""

    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def confusion_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual retained", "Actual churned"],
        columns=["Predicted retained", "Predicted churned"],
    )


def train_churn_model(data: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    """Train and evaluate the churn model on a stratified holdout set."""

    features = data[FEATURE_COLUMNS]
    labels = data[TARGET_COLUMN].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(test_x)[:, 1]
    metrics = metrics_at_threshold(test_y.to_numpy(), probabilities)

    importance = permutation_importance(
        pipeline,
        test_x,
        test_y,
        scoring="roc_auc",
        n_repeats=4,
        random_state=random_state,
        n_jobs=1,
    )
    importance_frame = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, probabilities)
    roc_points = pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )
    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        feature_importance=importance_frame,
        roc_points=roc_points,
        test_probabilities=probabilities,
        test_labels=test_y.to_numpy(),
    )


def score_players(pipeline: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Add churn probability and human-readable risk bands to player data."""

    scored = data.copy()
    scored["churn_risk"] = pipeline.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["risk_tier"] = pd.cut(
        scored["churn_risk"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
        ordered=True,
    ).astype(str)
    return scored
