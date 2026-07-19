"""Deterministic synthetic player telemetry for the GamePulse portfolio project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "churned_30d"
ID_COLUMNS = ["player_id", "observation_date", "region"]
CATEGORICAL_FEATURES = ["platform", "preferred_mode", "battle_pass_owned"]
NUMERIC_FEATURES = [
    "account_age_days",
    "sessions_30d",
    "avg_session_minutes",
    "days_since_last_login",
    "matches_30d",
    "win_rate",
    "levels_completed_90d",
    "social_connections",
    "purchases_90d",
    "spend_usd_90d",
    "support_tickets_90d",
    "crashes_30d",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def generate_player_data(n_players: int = 3_000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic but entirely synthetic player-level telemetry.

    Features represent activity observed before 2026-07-01. ``churned_30d``
    represents whether the synthetic player became inactive during the
    following 30 days. No real player or personal data is included.
    """

    if n_players < 100:
        raise ValueError("n_players must be at least 100")

    rng = np.random.default_rng(seed)
    player_ids = np.array([f"GP-{index:05d}" for index in range(1, n_players + 1)])
    platform = rng.choice(["PC", "Console", "Mobile"], n_players, p=[0.45, 0.34, 0.21])
    preferred_mode = rng.choice(
        ["Casual", "Ranked", "Co-op", "Battle Royale"],
        n_players,
        p=[0.38, 0.33, 0.17, 0.12],
    )
    region = rng.choice(
        ["APAC", "Europe", "North America", "LATAM", "MENA"],
        n_players,
        p=[0.31, 0.25, 0.23, 0.13, 0.08],
    )

    account_age_days = np.clip(rng.gamma(shape=2.3, scale=220, size=n_players), 21, 1_800).round().astype(int)
    latent_interest = rng.beta(2.4, 2.0, size=n_players)
    social_affinity = rng.beta(1.8, 2.5, size=n_players)

    platform_activity = np.select(
        [platform == "PC", platform == "Console", platform == "Mobile"],
        [1.10, 1.00, 0.88],
        default=1.0,
    )
    mode_activity = np.select(
        [preferred_mode == "Ranked", preferred_mode == "Co-op"],
        [1.16, 1.08],
        default=0.96,
    )

    session_rate = 3.0 + 23.0 * latent_interest * platform_activity * mode_activity
    sessions_30d = np.clip(rng.poisson(session_rate), 0, 70).astype(int)
    avg_session_minutes = np.clip(
        rng.normal(25 + 55 * latent_interest + 8 * (preferred_mode == "Ranked"), 15),
        5,
        180,
    ).round(1)

    inactivity_scale = np.maximum(1.6, 14 - 10 * latent_interest - 0.20 * sessions_30d)
    days_since_last_login = np.clip(rng.exponential(inactivity_scale), 0, 45).round().astype(int)
    matches_30d = np.clip(
        sessions_30d * rng.uniform(1.2, 3.4, n_players) + rng.normal(0, 4, n_players),
        0,
        220,
    ).round().astype(int)
    win_rate = np.clip(rng.beta(9, 9, n_players) + 0.04 * (preferred_mode == "Ranked"), 0.15, 0.85).round(3)
    levels_completed_90d = np.clip(
        rng.poisson(2 + sessions_30d * 0.42 + latent_interest * 5),
        0,
        70,
    ).astype(int)
    social_connections = np.clip(
        rng.poisson(1 + 12 * social_affinity + 0.12 * sessions_30d),
        0,
        75,
    ).astype(int)

    purchase_propensity = _sigmoid(-2.0 + 2.6 * latent_interest + 0.04 * sessions_30d)
    purchases_90d = np.where(
        rng.random(n_players) < purchase_propensity,
        rng.poisson(0.6 + 2.8 * purchase_propensity),
        0,
    ).astype(int)
    spend_usd_90d = np.where(
        purchases_90d > 0,
        rng.gamma(shape=1.7 + purchases_90d * 0.35, scale=6.8),
        0.0,
    )
    spend_usd_90d = np.clip(spend_usd_90d, 0, 250).round(2)
    battle_pass_probability = _sigmoid(-2.3 + 0.055 * sessions_30d + 0.055 * spend_usd_90d)
    battle_pass_owned = np.where(rng.random(n_players) < battle_pass_probability, "Yes", "No")

    support_tickets_90d = np.clip(
        rng.poisson(0.22 + 0.30 * (platform == "PC") + 0.10 * (sessions_30d > 30)),
        0,
        6,
    ).astype(int)
    crashes_30d = np.clip(
        rng.poisson(0.65 + 0.45 * (platform == "PC") + 0.12 * support_tickets_90d),
        0,
        10,
    ).astype(int)

    engagement_raw = (
        0.36 * np.minimum(sessions_30d / 35, 1)
        + 0.20 * np.minimum(avg_session_minutes / 100, 1)
        + 0.18 * np.minimum(levels_completed_90d / 30, 1)
        + 0.14 * np.minimum(social_connections / 25, 1)
        + 0.12 * (1 - np.minimum(days_since_last_login / 30, 1))
    )
    engagement_score = np.clip(engagement_raw * 100, 0, 100).round(1)

    churn_logit = (
        -0.75
        + 0.115 * (days_since_last_login - 6)
        - 0.052 * (sessions_30d - 15)
        - 0.012 * (avg_session_minutes - 50)
        - 0.025 * (levels_completed_90d - 10)
        - 0.025 * (social_connections - 8)
        + 0.125 * crashes_30d
        + 0.160 * support_tickets_90d
        - 0.32 * (battle_pass_owned == "Yes")
        + 0.15 * (preferred_mode == "Casual")
        + rng.normal(0, 0.70, n_players)
    )
    churn_probability = _sigmoid(churn_logit)
    churned_30d = (rng.random(n_players) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "player_id": player_ids,
            "observation_date": "2026-07-01",
            "platform": platform,
            "preferred_mode": preferred_mode,
            "region": region,
            "account_age_days": account_age_days,
            "sessions_30d": sessions_30d,
            "avg_session_minutes": avg_session_minutes,
            "days_since_last_login": days_since_last_login,
            "matches_30d": matches_30d,
            "win_rate": win_rate,
            "levels_completed_90d": levels_completed_90d,
            "social_connections": social_connections,
            "purchases_90d": purchases_90d,
            "spend_usd_90d": spend_usd_90d,
            "battle_pass_owned": battle_pass_owned,
            "support_tickets_90d": support_tickets_90d,
            "crashes_30d": crashes_30d,
            "engagement_score": engagement_score,
            TARGET_COLUMN: churned_30d,
        }
    )


def load_player_data(path: str | Path) -> pd.DataFrame:
    """Load and validate a GamePulse player dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Player data not found: {path}")
    data = pd.read_csv(path)
    required = set(ID_COLUMNS + CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["engagement_score", TARGET_COLUMN])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Player data is missing required columns: {', '.join(missing)}")
    if data["player_id"].duplicated().any():
        raise ValueError("player_id must be unique")
    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1")
    return data
