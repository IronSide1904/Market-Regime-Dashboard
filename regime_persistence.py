from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import REGIME_RULES


@dataclass(frozen=True)
class RegimePersistenceResult:
    available: bool
    current_regime: str = "Unavailable"
    days_in_current_regime: int = 0
    maturity: str = "Unavailable"
    score_cushion: float | None = None
    median_duration_current_regime: float | None = None
    transition_count: int = 0
    transition_frequency: float | None = None
    stability_score: int | None = None
    stability_read: str = "Regime persistence is unavailable."
    time_spent: dict[str, float] = field(default_factory=dict)
    episode_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)


def analyze_regime_persistence(scored: pd.DataFrame) -> RegimePersistenceResult:
    required_columns = {"Regime", "MR-1 Score"}
    if scored.empty or not required_columns.issubset(scored.columns):
        return RegimePersistenceResult(available=False)

    frame = scored.dropna(subset=["Regime", "MR-1 Score"]).copy()
    if frame.empty:
        return RegimePersistenceResult(available=False)

    episodes = _episodes(frame)
    current_regime = str(frame.iloc[-1]["Regime"])
    days_in_current_regime = _days_in_current_regime(frame)
    score_cushion = _score_cushion(float(frame.iloc[-1]["MR-1 Score"]), current_regime)
    episode_table = _episode_summary(episodes, total_days=len(frame))
    median_duration = _median_duration(episode_table, current_regime)
    time_spent = _time_spent(frame)
    transition_count = max(len(episodes) - 1, 0)
    transition_frequency = transition_count / max(len(frame) / 21, 1)
    stability_score = _stability_score(
        score_cushion=score_cushion,
        days_in_current_regime=days_in_current_regime,
        score_momentum=_score_momentum(frame),
        recent_transition_count=_recent_transition_count(episodes, frame.index[-1]),
    )
    maturity = _maturity_label(days_in_current_regime)

    return RegimePersistenceResult(
        available=True,
        current_regime=current_regime,
        days_in_current_regime=days_in_current_regime,
        maturity=maturity,
        score_cushion=score_cushion,
        median_duration_current_regime=median_duration,
        transition_count=transition_count,
        transition_frequency=transition_frequency,
        stability_score=stability_score,
        stability_read=_stability_read(
            current_regime=current_regime,
            days=days_in_current_regime,
            maturity=maturity,
            cushion=score_cushion,
            median_duration=median_duration,
            stability_score=stability_score,
        ),
        time_spent=time_spent,
        episode_table=episode_table,
        timeline=episodes,
    )


def build_peer_persistence_table(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, scored in histories.items():
        result = analyze_regime_persistence(scored)
        if not result.available:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Current Regime": result.current_regime,
                "Days in Regime": result.days_in_current_regime,
                "Score Cushion": result.score_cushion,
                "Time Risk-On %": result.time_spent.get("Risk-On", np.nan),
                "Transition Count": result.transition_count,
                "Stability": result.stability_score,
            }
        )
    return pd.DataFrame(rows)


def _episodes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start = frame.index[0]
    regime = str(frame.iloc[0]["Regime"])
    count = 1
    previous_idx = frame.index[0]
    for idx, row in frame.iloc[1:].iterrows():
        next_regime = str(row["Regime"])
        if next_regime == regime:
            count += 1
            previous_idx = idx
            continue
        rows.append({"Regime": regime, "Start": start, "End": previous_idx, "Days": count})
        start = idx
        regime = next_regime
        count = 1
        previous_idx = idx
    rows.append({"Regime": regime, "Start": start, "End": frame.index[-1], "Days": count})
    return pd.DataFrame(rows)


def _episode_summary(episodes: pd.DataFrame, total_days: int) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    summary = (
        episodes.groupby("Regime")["Days"]
        .agg(["mean", "median", "max", "count", "sum"])
        .rename(
            columns={
                "mean": "Avg Duration",
                "median": "Median Duration",
                "max": "Max Duration",
                "count": "Episodes",
                "sum": "Days",
            }
        )
        .reset_index()
    )
    summary["Time Spent %"] = summary["Days"] / max(total_days, 1)
    return summary[["Regime", "Avg Duration", "Median Duration", "Max Duration", "Episodes", "Time Spent %"]]


def _time_spent(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["Regime"].value_counts(normalize=True)
    return {regime: float(counts.get(regime, 0)) for regime in ["Risk-On", "Neutral", "Defensive"]}


def _days_in_current_regime(frame: pd.DataFrame) -> int:
    current_regime = frame.iloc[-1]["Regime"]
    days = 0
    for regime in reversed(frame["Regime"].tolist()):
        if regime != current_regime:
            break
        days += 1
    return days


def _score_cushion(score: float, regime: str) -> float:
    if regime == "Risk-On":
        return score - REGIME_RULES["Risk-On"]["min_score"]
    if regime == "Neutral":
        return min(score - REGIME_RULES["Neutral"]["min_score"], REGIME_RULES["Risk-On"]["min_score"] - score)
    return REGIME_RULES["Neutral"]["min_score"] - score


def _maturity_label(days: int) -> str:
    if days <= 5:
        return "New regime"
    if days <= 15:
        return "Developing regime"
    if days <= 40:
        return "Established regime"
    return "Extended regime"


def _score_momentum(frame: pd.DataFrame, days: int = 5) -> float:
    if len(frame) <= days:
        return 0.0
    return float(frame.iloc[-1]["MR-1 Score"] - frame.iloc[-days - 1]["MR-1 Score"])


def _recent_transition_count(episodes: pd.DataFrame, latest_date: pd.Timestamp, window_days: int = 21) -> int:
    if episodes.empty or "Start" not in episodes.columns:
        return 0
    cutoff = latest_date - pd.Timedelta(days=window_days * 2)
    recent_starts = pd.to_datetime(episodes["Start"], errors="coerce")
    return int((recent_starts > cutoff).sum())


def _stability_score(
    score_cushion: float,
    days_in_current_regime: int,
    score_momentum: float,
    recent_transition_count: int,
) -> int:
    cushion_points = min(max(score_cushion, 0), 20) * 2.0
    duration_points = min(days_in_current_regime, 30) / 30 * 30
    momentum_points = 15 if score_momentum >= 5 else 10 if score_momentum >= 0 else 4
    transition_penalty = min(recent_transition_count * 7, 25)
    return int(max(0, min(100, cushion_points + duration_points + momentum_points + 20 - transition_penalty)))


def _median_duration(episode_table: pd.DataFrame, current_regime: str) -> float | None:
    if episode_table.empty:
        return None
    rows = episode_table[episode_table["Regime"] == current_regime]
    if rows.empty:
        return None
    return float(rows.iloc[0]["Median Duration"])


def _stability_read(
    current_regime: str,
    days: int,
    maturity: str,
    cushion: float,
    median_duration: float | None,
    stability_score: int,
) -> str:
    median_text = "no historical median yet" if median_duration is None else f"a historical median of {median_duration:.0f} days"
    if stability_score >= 70:
        tone = "stable"
    elif stability_score >= 45:
        tone = "developing but not deeply anchored"
    else:
        tone = "fragile"
    article = "an" if maturity[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return (
        f"The ticker has been in {current_regime} for {days} trading days. "
        f"This is {article} {maturity.lower()} with {median_text}; the score cushion is {cushion:+.0f} points, "
        f"so regime stability looks {tone}."
    )
