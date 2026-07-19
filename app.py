"""Streamlit dashboard for GamePulse player churn and engagement analytics."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics import assign_retention_segments, filter_players, segment_summary
from src.data import load_player_data
from src.modeling import confusion_at_threshold, metrics_at_threshold, score_players, train_churn_model


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "gamepulse_players.csv.gz"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = ["#4ADE80", "#FBBF24", "#FB7185"]


st.set_page_config(
    page_title="GamePulse | Player Churn Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #070B18 0%, #0C1328 58%, #111A34 100%); }
    [data-testid="stSidebar"] { background: #0A1022; border-right: 1px solid #26345B; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(28,40,78,.96), rgba(18,27,56,.96));
        border: 1px solid #314475;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 108px;
    }
    [data-testid="stMetricValue"] { color: #F8FAFC; }
    [data-testid="stMetricLabel"] { color: #A5B4FC; }
    .gamepulse-hero {
        padding: 20px 24px;
        border: 1px solid #33477D;
        border-radius: 18px;
        background: linear-gradient(105deg, rgba(92,55,240,.28), rgba(14,165,233,.17));
        margin-bottom: 18px;
    }
    .gamepulse-hero h1 { margin: 0; color: #F8FAFC; font-size: 2.25rem; }
    .gamepulse-hero p { margin: 8px 0 0; color: #C7D2FE; }
    .synthetic-note {
        border-left: 4px solid #8B5CF6;
        background: rgba(88, 28, 135, .20);
        padding: 10px 14px;
        border-radius: 8px;
        color: #DDD6FE;
        margin: 0 0 16px 0;
    }
    div[data-baseweb="tab-list"] { gap: 14px; }
    button[data-baseweb="tab"] { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_player_data(DATA_PATH)


@st.cache_data
def load_training_result():
    return train_churn_model(load_player_data(DATA_PATH))


data = load_dataset()
training = load_training_result()
scored = assign_retention_segments(score_players(training.pipeline, data))

st.markdown(
    """
    <div class="gamepulse-hero">
      <h1>🎮 GamePulse</h1>
      <p>Player churn prediction, engagement intelligence, and retention action planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="synthetic-note">
      <strong>Portfolio demonstration:</strong> this dashboard uses deterministic synthetic telemetry, not real player or personal data.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Player filters")
all_platforms = sorted(scored["platform"].unique().tolist())
all_modes = sorted(scored["preferred_mode"].unique().tolist())
all_regions = sorted(scored["region"].unique().tolist())

platforms = st.sidebar.multiselect("Platforms", all_platforms, default=all_platforms)
modes = st.sidebar.multiselect("Preferred modes", all_modes, default=all_modes)
regions = st.sidebar.multiselect("Regions", all_regions, default=all_regions)
minimum_risk = st.sidebar.slider(
    "Minimum predicted churn risk",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    format="%.0f%%",
)
st.sidebar.caption("Risk represents the model's estimated probability of 30-day churn.")

filtered = filter_players(scored, platforms, modes, regions, minimum_risk)
if filtered.empty:
    st.warning("No players match these filters. Broaden at least one selection in the sidebar.")
    st.stop()

high_risk = filtered[filtered["risk_tier"] == "High"]
revenue_at_risk = high_risk["spend_usd_90d"].sum()
metric_columns = st.columns(5)
metric_columns[0].metric("Players", f"{len(filtered):,}")
metric_columns[1].metric("Observed 30-day churn", f"{filtered['churned_30d'].mean():.1%}")
metric_columns[2].metric("High-risk players", f"{len(high_risk):,}")
metric_columns[3].metric("Average engagement", f"{filtered['engagement_score'].mean():.1f}/100")
metric_columns[4].metric("90-day spend at risk", f"${revenue_at_risk:,.0f}")

overview_tab, explorer_tab, retention_tab, model_tab = st.tabs(
    ["Overview", "Player explorer", "Retention actions", "Model performance"]
)

with overview_tab:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Churn by platform")
        platform_summary = (
            filtered.groupby("platform", as_index=False)
            .agg(
                observed_churn=("churned_30d", "mean"),
                predicted_risk=("churn_risk", "mean"),
                players=("player_id", "count"),
            )
        )
        platform_long = platform_summary.melt(
            id_vars=["platform", "players"],
            value_vars=["observed_churn", "predicted_risk"],
            var_name="measure",
            value_name="rate",
        )
        platform_long["measure"] = platform_long["measure"].map(
            {"observed_churn": "Observed churn", "predicted_risk": "Average predicted risk"}
        )
        platform_chart = (
            alt.Chart(platform_long)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("platform:N", title=None, sort="-y"),
                y=alt.Y("rate:Q", title="Rate", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "measure:N",
                    title=None,
                    scale=alt.Scale(domain=["Observed churn", "Average predicted risk"], range=["#8B5CF6", "#22D3EE"]),
                ),
                xOffset="measure:N",
                tooltip=["platform:N", "measure:N", alt.Tooltip("rate:Q", format=".1%"), "players:Q"],
            )
            .properties(height=330)
        )
        st.altair_chart(platform_chart, use_container_width=True)

    with right:
        st.subheader("Predicted risk mix")
        risk_counts = filtered["risk_tier"].value_counts().reindex(RISK_ORDER, fill_value=0).rename_axis("risk_tier").reset_index(name="players")
        risk_chart = (
            alt.Chart(risk_counts)
            .mark_arc(innerRadius=72, outerRadius=132)
            .encode(
                theta=alt.Theta("players:Q"),
                color=alt.Color(
                    "risk_tier:N",
                    title="Risk tier",
                    sort=RISK_ORDER,
                    scale=alt.Scale(domain=RISK_ORDER, range=RISK_COLORS),
                ),
                tooltip=["risk_tier:N", "players:Q"],
            )
            .properties(height=330)
        )
        st.altair_chart(risk_chart, use_container_width=True)

    st.subheader("Engagement by preferred mode")
    mode_summary = (
        filtered.groupby("preferred_mode", as_index=False)
        .agg(average_engagement=("engagement_score", "mean"), players=("player_id", "count"))
        .sort_values("average_engagement", ascending=False)
    )
    mode_chart = (
        alt.Chart(mode_summary)
        .mark_bar(cornerRadiusEnd=6, color="#6366F1")
        .encode(
            x=alt.X("average_engagement:Q", title="Average engagement score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("preferred_mode:N", title=None, sort="-x"),
            tooltip=["preferred_mode:N", alt.Tooltip("average_engagement:Q", format=".1f"), "players:Q"],
        )
        .properties(height=230)
    )
    st.altair_chart(mode_chart, use_container_width=True)

with explorer_tab:
    st.subheader("Engagement versus churn risk")
    sample = filtered.sample(min(len(filtered), 1_500), random_state=42)
    scatter = (
        alt.Chart(sample)
        .mark_circle(opacity=0.68, stroke="#111827", strokeWidth=0.4)
        .encode(
            x=alt.X("engagement_score:Q", title="Engagement score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("churn_risk:Q", title="Predicted churn risk", axis=alt.Axis(format="%")),
            color=alt.Color(
                "risk_tier:N",
                title="Risk tier",
                sort=RISK_ORDER,
                scale=alt.Scale(domain=RISK_ORDER, range=RISK_COLORS),
            ),
            size=alt.Size("spend_usd_90d:Q", title="90-day spend", scale=alt.Scale(range=[35, 450])),
            tooltip=[
                "player_id:N",
                "platform:N",
                "preferred_mode:N",
                alt.Tooltip("engagement_score:Q", format=".1f"),
                alt.Tooltip("churn_risk:Q", format=".1%"),
                alt.Tooltip("spend_usd_90d:Q", format="$.2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.subheader("Prioritized player review")
    review_columns = [
        "player_id",
        "platform",
        "preferred_mode",
        "engagement_score",
        "churn_risk",
        "risk_tier",
        "spend_usd_90d",
        "days_since_last_login",
        "retention_segment",
    ]
    review = filtered.sort_values(["churn_risk", "spend_usd_90d"], ascending=[False, False])[review_columns].head(250)
    review_display = review.copy()
    review_display["churn_risk"] = review_display["churn_risk"] * 100
    st.dataframe(
        review_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_risk": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=100, format="%.1f%%"),
            "spend_usd_90d": st.column_config.NumberColumn("90-day spend", format="$%.2f"),
            "engagement_score": st.column_config.NumberColumn("Engagement", format="%.1f"),
        },
    )
    st.download_button(
        "Download filtered player review",
        data=review.to_csv(index=False).encode("utf-8"),
        file_name="gamepulse_player_review.csv",
        mime="text/csv",
    )

with retention_tab:
    st.subheader("Action-oriented retention segments")
    summary = segment_summary(filtered)
    segment_chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("players:Q", title="Players"),
            y=alt.Y("retention_segment:N", title=None, sort="-x"),
            color=alt.Color("average_risk:Q", title="Average risk", scale=alt.Scale(scheme="plasma")),
            tooltip=[
                "retention_segment:N",
                "players:Q",
                alt.Tooltip("share:Q", format=".1%"),
                alt.Tooltip("average_risk:Q", format=".1%"),
                alt.Tooltip("average_spend:Q", format="$.2f"),
            ],
        )
        .properties(height=310)
    )
    st.altair_chart(segment_chart, use_container_width=True)

    display_summary = summary.copy()
    display_summary["share"] = display_summary["share"].map(lambda value: f"{value:.1%}")
    display_summary["average_risk"] = display_summary["average_risk"].map(lambda value: f"{value:.1%}")
    display_summary["average_engagement"] = display_summary["average_engagement"].map(lambda value: f"{value:.1f}")
    display_summary["average_spend"] = display_summary["average_spend"].map(lambda value: f"${value:.2f}")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    action_queue = filtered[
        [
            "player_id",
            "retention_segment",
            "recommended_action",
            "churn_risk",
            "spend_usd_90d",
            "engagement_score",
        ]
    ].sort_values(["churn_risk", "spend_usd_90d"], ascending=[False, False])
    st.download_button(
        "Download retention action queue",
        data=action_queue.to_csv(index=False).encode("utf-8"),
        file_name="gamepulse_retention_actions.csv",
        mime="text/csv",
    )

with model_tab:
    threshold = st.slider(
        "Classification threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Lower thresholds identify more at-risk players but can create more false positives.",
    )
    threshold_metrics = metrics_at_threshold(training.test_labels, training.test_probabilities, threshold)
    model_metrics = st.columns(5)
    model_metrics[0].metric("ROC AUC", f"{threshold_metrics['roc_auc']:.3f}")
    model_metrics[1].metric("Accuracy", f"{threshold_metrics['accuracy']:.1%}")
    model_metrics[2].metric("Precision", f"{threshold_metrics['precision']:.1%}")
    model_metrics[3].metric("Recall", f"{threshold_metrics['recall']:.1%}")
    model_metrics[4].metric("F1 score", f"{threshold_metrics['f1']:.3f}")

    importance_column, roc_column = st.columns(2)
    with importance_column:
        st.subheader("Top churn drivers")
        top_importance = training.feature_importance.head(10).sort_values("importance")
        importance_chart = (
            alt.Chart(top_importance)
            .mark_bar(cornerRadiusEnd=6, color="#A78BFA")
            .encode(
                x=alt.X("importance:Q", title="Decrease in holdout ROC AUC"),
                y=alt.Y("feature:N", title=None, sort=None),
                tooltip=["feature:N", alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=360)
        )
        st.altair_chart(importance_chart, use_container_width=True)

    with roc_column:
        st.subheader("Holdout ROC curve")
        roc_chart = (
            alt.Chart(training.roc_points)
            .mark_line(color="#22D3EE", strokeWidth=3)
            .encode(
                x=alt.X("false_positive_rate:Q", title="False-positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("true_positive_rate:Q", title="True-positive rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    alt.Tooltip("false_positive_rate:Q", format=".3f"),
                    alt.Tooltip("true_positive_rate:Q", format=".3f"),
                ],
            )
            .properties(height=360)
        )
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#64748B", strokeDash=[6, 6]).encode(x="x:Q", y="y:Q")
        st.altair_chart(roc_chart + diagonal, use_container_width=True)

    st.subheader("Confusion matrix")
    confusion = confusion_at_threshold(training.test_labels, training.test_probabilities, threshold)
    confusion_long = confusion.rename_axis("actual").reset_index().melt(id_vars="actual", var_name="predicted", value_name="players")
    confusion_chart = (
        alt.Chart(confusion_long)
        .mark_rect(cornerRadius=5)
        .encode(
            x=alt.X("predicted:N", title=None),
            y=alt.Y("actual:N", title=None),
            color=alt.Color("players:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["actual:N", "predicted:N", "players:Q"],
        )
        .properties(height=230)
    )
    labels = confusion_chart.mark_text(fontSize=18, fontWeight="bold", color="#F8FAFC").encode(text="players:Q")
    st.altair_chart(confusion_chart + labels, use_container_width=True)

with st.expander("Methodology and responsible use"):
    st.markdown(
        """
        - **Target:** whether a synthetic player becomes inactive during the 30 days after the observation date.
        - **Model:** Random Forest with imputation and one-hot encoding, evaluated on a stratified 25% holdout set.
        - **Data:** deterministic synthetic telemetry created for education; no real users, identifiers, or proprietary game data.
        - **Intended use:** portfolio demonstration and retention-analysis practice only.
        - **Limitation:** performance on generated data does not represent production performance on a real game.
        """
    )
