from __future__ import annotations

import numpy as np
import pandas as pd

from config import RELATIVE_TREND_QUALITY_CONFIG, RELATIVE_TREND_TIMEFRAME_WINDOWS


def calculate_relative_strength_ratio(ticker_df, benchmark_df) -> pd.Series:
    ticker, benchmark = _align_close(ticker_df, benchmark_df)
    if ticker.empty or benchmark.empty:
        return pd.Series(dtype="float64")
    return (ticker / benchmark.replace(0, np.nan)).dropna()


def score_relative_trend_strength(
    ticker_df,
    benchmark_df,
    timeframe,
    config=None,
) -> dict:
    settings = config or RELATIVE_TREND_QUALITY_CONFIG
    weight = int(settings["weights"]["relative_trend_strength"])
    windows = _timeframe_windows(timeframe)
    ratio = calculate_relative_strength_ratio(ticker_df, benchmark_df)
    ticker_close, benchmark_close = _align_close(ticker_df, benchmark_df)
    warnings: list[str] = []
    drivers: list[str] = []

    if ratio.empty or ticker_close.empty or benchmark_close.empty:
        return _component("Relative trend data is unavailable.", 0, weight, warnings=["Missing ticker or benchmark data."])

    if _same_asset(ticker_df, benchmark_df):
        return _component(
            "Ticker equals benchmark; relative trend is neutral.",
            int(round(weight * 0.5)),
            weight,
            warnings=["Ticker and benchmark are the same, so outperformance is not meaningful."],
            details={"relative_trend_is_positive": False, "ratio": ratio},
        )

    lookback = min(int(windows["rs_lookback"]), len(ratio) - 1)
    ma_window = min(int(windows["rs_ma"]), len(ratio))
    if lookback < 2 or ma_window < 2:
        return _component("Insufficient data for selected timeframe.", 0, weight, warnings=["Not enough aligned history."])

    ratio_ma = ratio.rolling(ma_window, min_periods=max(2, ma_window // 2)).mean()
    latest_ratio = _valid_float(ratio.iloc[-1])
    latest_ma = _valid_float(ratio_ma.iloc[-1])
    ratio_base = _valid_float(ratio.iloc[-lookback - 1]) if len(ratio) > lookback else None
    ticker_return = _lookback_return(ticker_close, lookback)
    benchmark_return = _lookback_return(benchmark_close, lookback)
    excess_return = None if ticker_return is None or benchmark_return is None else ticker_return - benchmark_return

    above_ma = latest_ratio is not None and latest_ma is not None and latest_ratio > latest_ma
    rising_ratio = latest_ratio is not None and ratio_base is not None and latest_ratio > ratio_base
    outperforming = excess_return is not None and excess_return > 0
    positives = sum([above_ma, rising_ratio, outperforming])

    score = weight if positives == 3 else 25 if positives == 2 else 10 if positives == 1 else 0
    score = min(score, weight)

    if outperforming:
        drivers.append("Ticker is outperforming benchmark")
    if above_ma:
        drivers.append("Relative strength ratio is above its average")
    if rising_ratio:
        drivers.append("Relative strength ratio is rising")
    if positives == 0:
        warnings.append("Ticker is not outperforming the benchmark over the selected timeframe.")

    return _component(
        "Relative trend strength scored.",
        score,
        weight,
        drivers=drivers,
        warnings=warnings,
        details={
            "relative_trend_is_positive": bool(positives >= 2 or outperforming),
            "ratio": ratio,
            "ratio_ma": ratio_ma,
            "ticker_return": ticker_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "rs_lookback": lookback,
            "rs_ma": ma_window,
        },
    )


def score_relationship_stability(
    ticker_df,
    benchmark_df,
    timeframe,
    asset_type=None,
    config=None,
) -> dict:
    settings = config or RELATIVE_TREND_QUALITY_CONFIG
    weight = int(settings["weights"]["relationship_stability"])
    thresholds = settings["corr_thresholds"]
    windows = _timeframe_windows(timeframe)
    ticker_close, benchmark_close = _align_close(ticker_df, benchmark_df)
    ratio = calculate_relative_strength_ratio(ticker_df, benchmark_df)
    warnings: list[str] = []
    drivers: list[str] = []

    if ticker_close.empty or benchmark_close.empty or ratio.empty:
        return _component("Relationship data is unavailable.", 0, weight, warnings=["Missing aligned relationship history."])

    if _same_asset(ticker_df, benchmark_df):
        return _component(
            "Ticker equals benchmark; relationship stability is not applicable.",
            int(round(weight * 0.5)),
            weight,
            warnings=["Relationship stability is marked neutral because ticker equals benchmark."],
        )

    short_window = min(int(windows["corr_short"]), len(ticker_close) - 1)
    long_window = min(int(windows["corr_long"]), len(ticker_close) - 1)
    if short_window < 5 or long_window < 10:
        return _component("Insufficient data for selected timeframe.", 0, weight, warnings=["Not enough return history for correlation."])

    ticker_returns = ticker_close.pct_change()
    benchmark_returns = benchmark_close.pct_change()
    corr_short_series = ticker_returns.rolling(short_window).corr(benchmark_returns)
    corr_long_series = ticker_returns.rolling(long_window).corr(benchmark_returns)
    corr_short = _valid_float(corr_short_series.dropna().iloc[-1]) if not corr_short_series.dropna().empty else None
    corr_long = _valid_float(corr_long_series.dropna().iloc[-1]) if not corr_long_series.dropna().empty else None
    corr_change = None if corr_short is None or corr_long is None else abs(corr_short - corr_long)
    ratio_volatility = _relative_ratio_volatility(ratio, short_window)

    strong = (
        corr_short is not None
        and corr_long is not None
        and corr_short > thresholds["strong"]
        and corr_change is not None
        and corr_change < thresholds["max_correlation_change"]
        and not _ratio_volatility_extreme(ratio_volatility)
    )
    moderate = corr_short is not None and corr_short > thresholds["moderate"] and not _ratio_volatility_extreme(ratio_volatility)

    if strong:
        score = weight
        drivers.append("Ticker/benchmark relationship is stable")
    elif moderate:
        score = 20
        drivers.append("Ticker/benchmark relationship is mostly stable")
        warnings.append("Correlation stability is moderate, not strong.")
    elif corr_short is not None and corr_short > 0:
        score = 10
        warnings.append("Relationship is positive but noisy.")
    else:
        score = 0
        warnings.append("Correlation has weakened or inverted.")

    if corr_change is not None and corr_change >= thresholds["max_correlation_change"]:
        warnings.append("Correlation changed sharply versus the longer window.")
    if _ratio_volatility_extreme(ratio_volatility):
        warnings.append("Relative strength ratio volatility is elevated.")
    if _asset_type_warning(asset_type):
        warnings.append("Relationship stability may be less meaningful for this asset type.")

    return _component(
        "Relationship stability scored.",
        min(score, weight),
        weight,
        drivers=drivers,
        warnings=warnings,
        details={
            "corr_short": corr_short,
            "corr_long": corr_long,
            "corr_change": corr_change,
            "corr_short_window": short_window,
            "corr_long_window": long_window,
            "corr_short_series": corr_short_series,
            "ratio_volatility": ratio_volatility,
        },
    )


def score_volume_confirmation(
    volume_context,
    relative_trend_is_positive,
    config=None,
) -> dict:
    settings = config or RELATIVE_TREND_QUALITY_CONFIG
    weight = int(settings["weights"]["volume_confirmation"])
    context = str(_get_metric(volume_context, "context", "volume_context", "Volume Context") or "Unavailable")
    adjustment = _valid_float(_get_metric(volume_context, "adjustment", "volume_adjustment", "Volume Adjustment"))
    rvol = _valid_float(_get_metric(volume_context, "rvol_medium", "rvol_20d", "RVOL Medium", "RVOL 20D"))
    percentile = _valid_float(_get_metric(volume_context, "volume_percentile", "volume_percentile_1y", "Volume Percentile", "Volume Percentile 1Y"))
    daily_return = _valid_float(_get_metric(volume_context, "daily_return", "Ticker Return", "Daily Return"))
    drivers: list[str] = []
    warnings: list[str] = []

    if context in {"Accumulation", "Breakout Confirmation"} and relative_trend_is_positive:
        score = weight
        drivers.append("Volume confirms the relative trend")
    elif context == "Neutral" and _healthy_rvol(rvol, percentile):
        score = 20
        drivers.append("Volume participation is healthy")
    elif context == "Weak Participation":
        score = 10
        warnings.append("Price action lacks strong volume confirmation.")
    elif context in {"Distribution", "Panic / Liquidation"}:
        score = 0
        warnings.append("Volume is warning against the trend.")
    elif adjustment is not None and adjustment > 0 and relative_trend_is_positive:
        score = 20
        drivers.append("Volume adjustment supports the setup")
    elif adjustment is not None and adjustment < 0:
        score = 5
        warnings.append("Volume adjustment is negative.")
    elif _healthy_rvol(rvol, percentile) and relative_trend_is_positive:
        score = 20
        drivers.append("Relative volume is healthy")
    else:
        score = 10 if relative_trend_is_positive else 0
        warnings.append("Volume does not clearly confirm the relative trend.")

    if daily_return is not None and daily_return < 0 and context in {"Distribution", "Panic / Liquidation"}:
        score = 0
        warnings.append("High-volume downside move contradicts the setup.")

    return _component(
        "Volume confirmation scored.",
        min(score, weight),
        weight,
        drivers=drivers,
        warnings=warnings,
        details={"context": context, "adjustment": adjustment, "rvol": rvol, "percentile": percentile},
    )


def calculate_clean_relative_trend_score(
    ticker_df,
    benchmark_df,
    timeframe,
    volume_context=None,
    asset_type=None,
    config=None,
) -> dict:
    settings = config or RELATIVE_TREND_QUALITY_CONFIG
    if not settings.get("enabled", True):
        return _empty_result("Clean Relative Trend Score is disabled.")

    trend = score_relative_trend_strength(ticker_df, benchmark_df, timeframe, config=settings)
    stability = score_relationship_stability(ticker_df, benchmark_df, timeframe, asset_type=asset_type, config=settings)
    volume = score_volume_confirmation(
        volume_context=volume_context,
        relative_trend_is_positive=bool(trend.get("details", {}).get("relative_trend_is_positive")),
        config=settings,
    )
    score = int(np.clip(trend["score"] + stability["score"] + volume["score"], 0, 100))
    label = _label(score, settings)
    drivers = [*trend["positive_drivers"], *stability["positive_drivers"], *volume["positive_drivers"]]
    warnings = _dedupe([*trend["warnings"], *stability["warnings"], *volume["warnings"]])
    explanation = _explanation(score, label, trend, stability, volume)
    history = _history_frame(trend, stability)

    return {
        "available": True,
        "score": score,
        "label": label,
        "relative_trend_score": int(trend["score"]),
        "relationship_stability_score": int(stability["score"]),
        "volume_confirmation_score": int(volume["score"]),
        "relative_trend_max": int(trend["max_score"]),
        "relationship_stability_max": int(stability["max_score"]),
        "volume_confirmation_max": int(volume["max_score"]),
        "status": _status(label),
        "positive_drivers": drivers,
        "warnings": warnings,
        "explanation": explanation,
        "timeframe": timeframe,
        "details": {
            "relative_trend": trend.get("details", {}),
            "relationship_stability": stability.get("details", {}),
            "volume_confirmation": volume.get("details", {}),
        },
        "history": history,
    }


def _history_frame(trend: dict, stability: dict) -> pd.DataFrame:
    ratio = trend.get("details", {}).get("ratio")
    if not isinstance(ratio, pd.Series) or ratio.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(index=ratio.index)
    frame["Relative Ratio"] = ratio
    ratio_ma = trend.get("details", {}).get("ratio_ma")
    if isinstance(ratio_ma, pd.Series):
        frame["Relative Ratio MA"] = ratio_ma.reindex(frame.index)
    corr_short_window = stability.get("details", {}).get("corr_short_window")
    corr_short_series = stability.get("details", {}).get("corr_short_series")
    if isinstance(corr_short_series, pd.Series):
        frame["Rolling Correlation"] = corr_short_series.reindex(frame.index)
    elif corr_short_window:
        frame["Rolling Correlation"] = np.nan
    return frame


def _align_close(ticker_df, benchmark_df) -> tuple[pd.Series, pd.Series]:
    ticker = _close_series(ticker_df)
    benchmark = _close_series(benchmark_df)
    if ticker.empty or benchmark.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    common = ticker.index.intersection(benchmark.index)
    ticker = ticker.loc[common].dropna()
    benchmark = benchmark.loc[common].dropna()
    common = ticker.index.intersection(benchmark.index)
    return ticker.loc[common], benchmark.loc[common]


def _close_series(frame) -> pd.Series:
    if frame is None:
        return pd.Series(dtype="float64")
    if isinstance(frame, pd.Series):
        series = frame.copy()
    elif isinstance(frame, pd.DataFrame):
        if "Close" in frame.columns:
            series = frame["Close"]
        elif "Ticker Close" in frame.columns:
            series = frame["Ticker Close"]
        elif "Benchmark Close" in frame.columns:
            series = frame["Benchmark Close"]
        else:
            return pd.Series(dtype="float64")
    else:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index)
    return series.dropna()


def _same_asset(ticker_df, benchmark_df) -> bool:
    ticker = getattr(ticker_df, "attrs", {}).get("symbol")
    benchmark = getattr(benchmark_df, "attrs", {}).get("symbol")
    return bool(ticker and benchmark and str(ticker).upper() == str(benchmark).upper())


def _timeframe_windows(timeframe: str) -> dict:
    return RELATIVE_TREND_TIMEFRAME_WINDOWS.get(timeframe, RELATIVE_TREND_TIMEFRAME_WINDOWS["1M"])


def _lookback_return(series: pd.Series, days: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= days:
        return None
    base = clean.iloc[-days - 1]
    if base == 0 or pd.isna(base):
        return None
    return _valid_float(clean.iloc[-1] / base - 1)


def _relative_ratio_volatility(ratio: pd.Series, window: int) -> float | None:
    returns = ratio.pct_change().dropna()
    if len(returns) < max(5, min(window, 10)):
        return None
    return _valid_float(returns.tail(window).std() * np.sqrt(252))


def _ratio_volatility_extreme(value) -> bool:
    number = _valid_float(value)
    return bool(number is not None and number > 0.55)


def _asset_type_warning(asset_type) -> bool:
    value = str(asset_type or "").lower()
    return any(token in value for token in ["crypto", "inverse", "leveraged", "low-correlation"])


def _healthy_rvol(rvol, percentile) -> bool:
    rv = _valid_float(rvol)
    pct = _valid_float(percentile)
    return bool(rv is not None and rv >= 1.1 and (pct is None or pct >= 60))


def _component(message: str, score: int, max_score: int, drivers=None, warnings=None, details=None) -> dict:
    return {
        "message": message,
        "score": int(max(0, min(max_score, score))),
        "max_score": int(max_score),
        "positive_drivers": drivers or [],
        "warnings": warnings or [],
        "details": details or {},
    }


def _empty_result(warning: str) -> dict:
    return {
        "available": False,
        "score": 0,
        "label": "Unavailable",
        "relative_trend_score": 0,
        "relationship_stability_score": 0,
        "volume_confirmation_score": 0,
        "relative_trend_max": 40,
        "relationship_stability_max": 30,
        "volume_confirmation_max": 30,
        "status": "Unavailable",
        "positive_drivers": [],
        "warnings": [warning],
        "explanation": warning,
        "history": pd.DataFrame(),
        "details": {},
    }


def _label(score: int, config: dict) -> str:
    labels = config["labels"]
    if score >= labels["clean"]:
        return "Clean Trend"
    if score >= labels["good_but_watch"]:
        return "Good but Watch"
    if score >= labels["mixed"]:
        return "Mixed / Noisy"
    return "Weak / Unconfirmed"


def _status(label: str) -> str:
    if label == "Clean Trend":
        return "Confirmed relative trend"
    if label == "Good but Watch":
        return "Positive but imperfect"
    if label == "Mixed / Noisy":
        return "Noisy setup"
    return "Unconfirmed trend"


def _explanation(score: int, label: str, trend: dict, stability: dict, volume: dict) -> str:
    if score >= 80:
        return (
            "The ticker is outperforming the benchmark with a rising relative-strength profile, "
            "the benchmark relationship is stable enough to trust, and volume is confirming the move."
        )
    if score >= 60:
        return (
            "The ticker has a positive relative setup, but at least one confirmation layer is imperfect. "
            "Use it as a tradable setup only with normal risk controls."
        )
    if score >= 40:
        return (
            "The setup is mixed: relative strength, relationship stability, and volume are not all aligned. "
            "Avoid treating one strong metric as enough confirmation."
        )
    return (
        "The setup is noisy: relative strength is not confirmed by a stable relationship and volume does not support the move."
    )


def _get_metric(metrics, *keys):
    if metrics is None:
        return None
    if isinstance(metrics, pd.Series):
        for key in keys:
            if key in metrics:
                return metrics.get(key)
        return None
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics:
                return metrics.get(key)
    return None


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        if item and item not in output:
            output.append(item)
    return output


def _valid_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or np.isinf(number):
        return None
    return number
