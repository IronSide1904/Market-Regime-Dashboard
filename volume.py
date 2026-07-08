from __future__ import annotations

import numpy as np
import pandas as pd

from config import VOLUME_CONFIG


UNAVAILABLE_CONTEXT = {
    "context": "Unavailable",
    "adjustment": 0,
    "status": "Volume unavailable",
    "rvol_20d": np.nan,
    "rvol_50d": np.nan,
    "volume_percentile_1y": np.nan,
    "volume_z_score": np.nan,
    "dollar_volume": np.nan,
    "daily_float_turnover": np.nan,
    "five_day_float_turnover": np.nan,
    "explanation": "Volume context is unavailable because usable ticker volume data was not found.",
}


def calculate_volume_metrics(price_df: pd.DataFrame, shares_float: float | None = None) -> pd.DataFrame:
    data = price_df.copy()
    if "Volume" not in data.columns:
        return pd.DataFrame(index=price_df.index)

    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    data["Avg Volume 20D"] = data["Volume"].rolling(20).mean()
    data["Avg Volume 50D"] = data["Volume"].rolling(50).mean()
    data["Relative Volume 20D"] = data["Volume"] / data["Avg Volume 20D"]
    data["Relative Volume 50D"] = data["Volume"] / data["Avg Volume 50D"]
    data["Volume Percentile 1Y"] = data["Volume"].rolling(252).apply(_percentile_rank, raw=False)

    volume_std = data["Volume"].rolling(252).std()
    volume_mean = data["Volume"].rolling(252).mean()
    data["Volume Z-Score"] = (data["Volume"] - volume_mean) / volume_std
    data["Dollar Volume"] = data["Close"] * data["Volume"] if "Close" in data.columns else np.nan

    if shares_float:
        data["Daily Float Turnover"] = data["Volume"] / shares_float
        data["5D Float Turnover"] = data["Volume"].rolling(5).sum() / shares_float
    else:
        data["Daily Float Turnover"] = np.nan
        data["5D Float Turnover"] = np.nan

    return data


def classify_volume_context(
    price_df: pd.DataFrame,
    trend_status: bool,
    vix_status: bool,
    shares_float: float | None = None,
    config: dict | None = None,
) -> dict:
    settings = config or VOLUME_CONFIG
    validation_warnings = validate_volume_data(price_df)
    if validation_warnings:
        result = UNAVAILABLE_CONTEXT.copy()
        result["warnings"] = validation_warnings
        return result

    metrics = calculate_volume_metrics(price_df, shares_float=shares_float)
    if metrics.empty:
        result = UNAVAILABLE_CONTEXT.copy()
        result["warnings"] = ["Volume metrics could not be calculated."]
        return result

    latest = metrics.dropna(subset=["Close", "Volume"]).iloc[-1]
    previous_close = metrics["Close"].dropna().iloc[-2] if metrics["Close"].dropna().shape[0] >= 2 else np.nan
    daily_return = (latest["Close"] / previous_close - 1) if pd.notna(previous_close) and previous_close else 0.0
    previous = metrics.iloc[-2] if len(metrics) >= 2 else pd.Series(dtype=float)
    return _classify_metric_row(
        latest=latest,
        previous=previous,
        daily_return=daily_return,
        trend_status=trend_status,
        vix_status=vix_status,
        settings=settings,
        warnings=[],
    )


def classify_volume_history(
    price_df: pd.DataFrame,
    trend_status: pd.Series,
    vix_status: pd.Series,
    shares_float: float | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    metrics = calculate_volume_metrics(price_df, shares_float=shares_float)
    if metrics.empty:
        return pd.DataFrame(index=price_df.index)

    previous_close = metrics["Close"].shift(1)
    daily_returns = metrics["Close"] / previous_close - 1
    rows = []
    for position, idx in enumerate(metrics.index):
        if position < 251:
            rows.append(_history_row(UNAVAILABLE_CONTEXT))
            continue
        latest = metrics.loc[idx]
        previous = metrics.iloc[position - 1] if position > 0 else pd.Series(dtype=float)
        result = _classify_metric_row(
            latest=latest,
            previous=previous,
            daily_return=float(daily_returns.loc[idx]) if pd.notna(daily_returns.loc[idx]) else 0.0,
            trend_status=bool(trend_status.reindex(metrics.index).loc[idx]),
            vix_status=bool(vix_status.reindex(metrics.index).loc[idx]),
            settings=config or VOLUME_CONFIG,
            warnings=[],
        )
        rows.append(_history_row(result))

    history = pd.DataFrame(rows, index=metrics.index)
    for column in ["Volume", "Avg Volume 20D", "Avg Volume 50D"]:
        if column in metrics.columns:
            history[column] = metrics[column]
    return history


def _classify_metric_row(
    latest: pd.Series,
    previous: pd.Series,
    daily_return: float,
    trend_status: bool,
    vix_status: bool,
    settings: dict,
    warnings: list[str],
) -> dict:
    rvol_20d = float(latest.get("Relative Volume 20D", np.nan))
    volume_percentile = float(latest.get("Volume Percentile 1Y", np.nan))
    close = float(latest["Close"])
    sma_50 = float(latest.get("SMA 50D", np.nan))
    sma_200 = float(latest.get("SMA 200D", np.nan))
    previous_close = float(previous.get("Close", np.nan))
    previous_sma_200 = float(previous.get("SMA 200D", np.nan))

    is_high_volume = rvol_20d > settings["rvol_high"] and volume_percentile > settings["volume_percentile_high"]
    is_extreme_volume = rvol_20d > settings["rvol_extreme"] and volume_percentile > settings["volume_percentile_extreme"]
    above_50 = pd.notna(sma_50) and close > sma_50
    above_200 = trend_status if pd.isna(sma_200) else close > sma_200
    below_50_or_200 = (pd.notna(sma_50) and close < sma_50) or (pd.notna(sma_200) and close < sma_200)
    crossed_above_200 = pd.notna(previous_sma_200) and previous_close <= previous_sma_200 and pd.notna(sma_200) and close > sma_200

    context = "Neutral"
    adjustment = 0
    status = "Neutral"
    if is_extreme_volume and daily_return < settings["sharp_down_day_pct"] / 100 and not vix_status:
        context = "Panic / Liquidation"
        adjustment = settings["max_negative_adjustment"]
        status = "Bearish volume warning"
    elif is_high_volume and crossed_above_200 and daily_return > 0:
        context = "Breakout Confirmation"
        adjustment = settings["max_positive_adjustment"]
        status = "Bullish confirmation"
    elif is_high_volume and daily_return > 0 and above_200:
        context = "Accumulation"
        adjustment = settings["max_positive_adjustment"] if daily_return > 0.02 and above_50 else 10
        status = "Bullish confirmation"
    elif is_high_volume and daily_return < 0 and below_50_or_200:
        context = "Distribution"
        adjustment = settings["max_negative_adjustment"] if not above_200 else -10
        status = "Bearish volume warning"
    elif daily_return > 0 and rvol_20d < settings["weak_volume_rvol"]:
        context = "Weak Participation"
        adjustment = -5
        status = "Weak confirmation"

    adjustment = int(max(settings["max_negative_adjustment"], min(settings["max_positive_adjustment"], adjustment)))
    return {
        "context": context,
        "adjustment": adjustment,
        "status": status,
        "rvol_20d": rvol_20d,
        "rvol_50d": float(latest.get("Relative Volume 50D", np.nan)),
        "volume_percentile_1y": volume_percentile,
        "volume_z_score": float(latest.get("Volume Z-Score", np.nan)),
        "dollar_volume": float(latest.get("Dollar Volume", np.nan)),
        "daily_float_turnover": float(latest.get("Daily Float Turnover", np.nan)),
        "five_day_float_turnover": float(latest.get("5D Float Turnover", np.nan)),
        "explanation": _volume_explanation(
            context=context,
            adjustment=adjustment,
            rvol_20d=rvol_20d,
            volume_percentile=volume_percentile,
            daily_return=daily_return,
            above_200=above_200,
            daily_float_turnover=latest.get("Daily Float Turnover", np.nan),
        ),
        "warnings": warnings,
    }


def validate_volume_data(price_df: pd.DataFrame) -> list[str]:
    warnings = []
    if price_df.empty or "Volume" not in price_df.columns:
        return ["Volume column is missing."]

    volume = pd.to_numeric(price_df["Volume"], errors="coerce").dropna()
    if volume.empty:
        return ["Volume column is not numeric."]
    if (volume == 0).all():
        warnings.append("Volume is entirely zero.")
    if volume.shape[0] < 252:
        warnings.append("At least 252 trading days are required for volume context.")
    if pd.isna(pd.to_numeric(price_df["Volume"], errors="coerce").iloc[-1]):
        warnings.append("Current volume is unavailable.")
    return warnings


def _history_row(result: dict) -> dict:
    return {
        "Volume Context": result["context"],
        "Volume Status": result["status"],
        "Volume Adjustment": result["adjustment"],
        "RVOL 20D": result["rvol_20d"],
        "RVOL 50D": result["rvol_50d"],
        "Volume Percentile 1Y": result["volume_percentile_1y"],
        "Volume Z-Score": result["volume_z_score"],
        "Dollar Volume": result["dollar_volume"],
        "Daily Float Turnover": result["daily_float_turnover"],
        "5D Float Turnover": result["five_day_float_turnover"],
        "Volume Explanation": result["explanation"],
    }


def _percentile_rank(values: pd.Series) -> float:
    current = values.iloc[-1]
    if pd.isna(current):
        return np.nan
    return float((values <= current).mean() * 100)


def _volume_explanation(
    context: str,
    adjustment: int,
    rvol_20d: float,
    volume_percentile: float,
    daily_return: float,
    above_200: bool,
    daily_float_turnover: float,
) -> str:
    rvol_text = "N/A" if pd.isna(rvol_20d) else f"{rvol_20d:.1f}x"
    percentile_text = "N/A" if pd.isna(volume_percentile) else f"{volume_percentile:.0f}th percentile"
    direction = "rising" if daily_return > 0 else "falling" if daily_return < 0 else "flat"
    trend_text = "above" if above_200 else "below"
    float_text = (
        " Float turnover unavailable because shares float data was not found."
        if pd.isna(daily_float_turnover)
        else f" Daily volume equals {daily_float_turnover:.1%} of the float."
    )

    if context in {"Accumulation", "Breakout Confirmation"}:
        return (
            f"Volume is confirming the move. Current volume is {rvol_text} the 20-day average "
            f"and ranks in the {percentile_text} of the past year. Price is {direction} while "
            f"the ticker is {trend_text} its 200-day average, so volume adds {adjustment:+d} points."
            f"{float_text}"
        )
    if context in {"Distribution", "Panic / Liquidation"}:
        return (
            f"Volume is a warning. Current volume is {rvol_text} normal 20-day activity and ranks "
            f"in the {percentile_text} of the past year while price is {direction}. "
            f"That reduces the final MR-1 score by {abs(adjustment)} points.{float_text}"
        )
    if context == "Weak Participation":
        return (
            f"Price is rising, but volume is only {rvol_text} the 20-day average. "
            f"That weak participation reduces the final MR-1 score by {abs(adjustment)} points.{float_text}"
        )
    if context == "Unavailable":
        return UNAVAILABLE_CONTEXT["explanation"]
    return (
        f"Volume is not sending a strong signal today. Trading activity is {rvol_text} the 20-day "
        f"average and ranks in the {percentile_text} of the past year, so the score adjustment is 0."
        f"{float_text}"
    )
