"""Player segmentation and retention analytics for GamePulse."""

from __future__ import annotations

import pandas as pd


ACTION_MAP = {
    "High-value at risk": "Offer a personalized return reward and surface unfinished progression.",
    "New player friction": "Trigger onboarding help, easier starter challenges, and crash support.",
    "Socially disconnected": "Recommend communities, co-op groups, and friend-finder missions.",
    "Loyal core": "Invite to advanced events, feedback programs, and ambassador rewards.",
    "Monitor": "Continue standard live-ops messaging and monitor engagement changes.",
}


def assign_retention_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive action-oriented player segments."""

    result = scored.copy()
    positive_spend_median = result.loc[result["spend_usd_90d"] > 0, "spend_usd_90d"].median()
    if pd.isna(positive_spend_median):
        positive_spend_median = 0.0

    segment = pd.Series("Monitor", index=result.index, dtype="object")
    loyal = (result["engagement_score"] >= 70) & (result["churn_risk"] < 0.30)
    social = (result["social_connections"] <= 3) & (result["churn_risk"] >= 0.45)
    new_player = (
        (result["account_age_days"] <= 90)
        & (result["engagement_score"] < 45)
        & (result["churn_risk"] >= 0.45)
    )
    high_value = (
        (result["spend_usd_90d"] >= positive_spend_median)
        & (result["spend_usd_90d"] > 0)
        & (result["churn_risk"] >= 0.60)
    )

    segment.loc[loyal] = "Loyal core"
    segment.loc[social] = "Socially disconnected"
    segment.loc[new_player] = "New player friction"
    segment.loc[high_value] = "High-value at risk"
    result["retention_segment"] = segment
    result["recommended_action"] = result["retention_segment"].map(ACTION_MAP)
    return result


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment size, risk, engagement, spend, and action."""

    summary = (
        segmented.groupby("retention_segment", as_index=False, observed=True)
        .agg(
            players=("player_id", "count"),
            average_risk=("churn_risk", "mean"),
            average_engagement=("engagement_score", "mean"),
            average_spend=("spend_usd_90d", "mean"),
        )
    )
    summary["share"] = summary["players"] / len(segmented)
    summary["recommended_action"] = summary["retention_segment"].map(ACTION_MAP)
    return summary.sort_values(["average_risk", "players"], ascending=[False, False]).reset_index(drop=True)


def filter_players(
    data: pd.DataFrame,
    platforms: list[str],
    modes: list[str],
    regions: list[str],
    minimum_risk: float,
) -> pd.DataFrame:
    """Apply dashboard filters without mutating the scored dataset."""

    mask = (
        data["platform"].isin(platforms)
        & data["preferred_mode"].isin(modes)
        & data["region"].isin(regions)
        & (data["churn_risk"] >= minimum_risk)
    )
    return data.loc[mask].copy()
