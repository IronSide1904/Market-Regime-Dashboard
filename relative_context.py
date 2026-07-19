from __future__ import annotations

import numpy as np
import pandas as pd

from config import RELATIVE_CONTEXT_CONFIG


def unavailable_context(ticker: str, benchmark: str, warning: str = "Relative context unavailable.") -> dict:
    return {
        "available": False,
        "ticker": str(ticker or "").upper(),
        "benchmark": str(benchmark or "").upper(),
        "relative_5d": None,
        "relative_20d": None,
        "relative_60d": None,
        "relative_trend": "Unavailable",
        "relative_zscore": None,
        "relative_extension_label": "Unavailable",
        "correlation_20d": None,
        "correlation_60d": None,
        "correlation_120d": None,
        "correlation_trend": "Unavailable",
        "beta_20d": None,
        "beta_60d": None,
        "beta_trend": "Unavailable",
        "volume_confirmation": "Unavailable",
        "relationship_status": "Unavailable",
        "score_adjustment": 0,
        "confidence": "Low",
        "interpretation": "Relative context unavailable because price or benchmark data is missing.",
        "warnings": [warning],
        "history": pd.DataFrame(),
        "debug": {},
    }


def align_price_data(ticker_df, benchmark_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_clean = _standardize_ohlcv(ticker_df)
    benchmark_clean = _standardize_ohlcv(benchmark_df)
    if ticker_clean.empty or benchmark_clean.empty or "Close" not in ticker_clean.columns or "Close" not in benchmark_clean.columns:
        return pd.DataFrame(), pd.DataFrame()

    common_index = ticker_clean.index.intersection(benchmark_clean.index)
    ticker_aligned = ticker_clean.loc[common_index].copy()
    benchmark_aligned = benchmark_clean.loc[common_index].copy()
    valid = ticker_aligned["Close"].notna() & benchmark_aligned["Close"].notna()
    return ticker_aligned.loc[valid], benchmark_aligned.loc[valid]


def calculate_relative_ratio(ticker_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    benchmark = pd.to_numeric(benchmark_close, errors="coerce").replace(0, np.nan)
    ticker = pd.to_numeric(ticker_close, errors="coerce")
    return (ticker / benchmark).dropna()


def calculate_relative_performance(relative_ratio: pd.Series) -> dict:
    return {
        "relative_5d": _period_return(relative_ratio, 5),
        "relative_20d": _period_return(relative_ratio, 20),
        "relative_60d": _period_return(relative_ratio, 60),
    }


def calculate_rolling_correlation(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
    windows=(20, 60, 120),
) -> dict:
    output = {}
    for window in windows:
        series = ticker_returns.rolling(window).corr(benchmark_returns)
        output[f"correlation_{window}d"] = _valid_float(series.dropna().iloc[-1]) if not series.dropna().empty else None
    return output


def calculate_rolling_beta(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
    windows=(20, 60),
) -> dict:
    output = {}
    for window in windows:
        variance = benchmark_returns.rolling(window).var()
        covariance = ticker_returns.rolling(window).cov(benchmark_returns)
        beta = covariance / variance.replace(0, np.nan)
        output[f"beta_{window}d"] = _valid_float(beta.dropna().iloc[-1]) if not beta.dropna().empty else None
    return output


def calculate_relative_zscore(relative_ratio: pd.Series, window: int = 60) -> float | None:
    if relative_ratio.shape[0] < window:
        return None
    rolling_mean = relative_ratio.rolling(window).mean()
    rolling_std = relative_ratio.rolling(window).std()
    latest_std = _valid_float(rolling_std.iloc[-1])
    if latest_std is None or latest_std == 0:
        return None
    return _valid_float((relative_ratio.iloc[-1] - rolling_mean.iloc[-1]) / latest_std)


def classify_relative_trend(relative_5d, relative_20d, relative_60d) -> str:
    threshold = RELATIVE_CONTEXT_CONFIG["relative_trend_threshold"]
    value = _valid_float(relative_20d)
    if value is None:
        return "Unavailable"
    if value > threshold:
        return "Improving"
    if value < -threshold:
        return "Weakening"
    return "Flat"


def classify_correlation_stability(corr_20d, corr_60d, corr_120d) -> str:
    c20 = _valid_float(corr_20d)
    c60 = _valid_float(corr_60d)
    c120 = _valid_float(corr_120d)
    if c20 is None or c60 is None:
        return "Unavailable"
    if c20 < RELATIVE_CONTEXT_CONFIG["correlation_unstable_level"] or (c120 is not None and c20 < 0 < c120):
        return "Unstable"
    if c20 <= c60 - RELATIVE_CONTEXT_CONFIG["correlation_drop_warning"]:
        return "Weakening"
    if c60 > 0.50:
        return "Stable"
    return "Weakening"


def classify_beta_trend(beta_20d, beta_60d) -> str:
    b20 = _valid_float(beta_20d)
    b60 = _valid_float(beta_60d)
    if b20 is None or b60 is None:
        return "Unavailable"
    if abs(b20) > 4 or abs(b60) > 4:
        return "Unstable"
    threshold = RELATIVE_CONTEXT_CONFIG["beta_change_threshold"]
    if b20 > b60 + threshold:
        return "Rising"
    if b20 < b60 - threshold:
        return "Falling"
    return "Stable"


def classify_relative_extension(relative_zscore) -> str:
    zscore = _valid_float(relative_zscore)
    if zscore is None:
        return "Unavailable"
    strong = RELATIVE_CONTEXT_CONFIG["strong_extension_z"]
    moderate = RELATIVE_CONTEXT_CONFIG["moderate_extension_z"]
    if zscore >= strong:
        return "Strongly extended"
    if zscore >= moderate:
        return "Moderately extended"
    if zscore <= -strong:
        return "Deep weakness"
    if zscore <= -moderate:
        return "Underperforming"
    return "Neutral"


def classify_volume_confirmation(
    ticker_price_change,
    benchmark_price_change,
    relative_20d,
    ticker_volume_metrics,
    benchmark_volume_metrics,
) -> str:
    rel20 = _valid_float(relative_20d)
    ticker_change = _valid_float(ticker_price_change)
    if rel20 is None or ticker_change is None:
        return "Unavailable"

    ticker_rvol = _metric_value(ticker_volume_metrics, "rvol_20d", "RVOL 20D", "Relative Volume 20D")
    ticker_percentile = _metric_value(
        ticker_volume_metrics,
        "volume_percentile_1y",
        "Volume Percentile 1Y",
        "volume_percentile",
    )
    elevated_volume = (
        ticker_rvol is not None
        and ticker_rvol >= 1.2
        and (ticker_percentile is None or ticker_percentile >= 70)
    )
    weak_volume = ticker_rvol is not None and ticker_rvol < 0.8

    if rel20 > 0.03 and ticker_change > 0:
        return "Confirmed" if elevated_volume else "Not confirmed" if weak_volume else "Neutral"
    if rel20 < -0.03 and ticker_change < 0:
        return "Distribution warning" if elevated_volume else "Neutral"
    return "Neutral"


def classify_relationship_status(metrics: dict) -> dict:
    warnings: list[str] = []
    status = "Neutral"
    adjustment = 0
    confidence = "Medium"

    trend = metrics.get("relative_trend")
    extension = metrics.get("relative_extension_label")
    corr = metrics.get("correlation_trend")
    beta = metrics.get("beta_trend")
    volume = metrics.get("volume_confirmation")

    if trend == "Unavailable":
        return {
            "relationship_status": "Unavailable",
            "score_adjustment": 0,
            "confidence": "Low",
            "warnings": ["Relative context unavailable."],
        }

    if corr == "Unstable" or beta == "Unstable":
        status = "Broken"
        adjustment = -5
        confidence = "Low"
        warnings.append("Comparison confirmation is unreliable because the relationship is unstable.")
    elif trend == "Improving" and volume == "Confirmed" and extension != "Strongly extended":
        status = "Supportive"
        adjustment = 8 if corr == "Stable" else 5
        confidence = "High" if corr == "Stable" else "Medium"
    elif trend == "Weakening" and volume == "Distribution warning":
        status = "Warning"
        adjustment = -10
        confidence = "High"
        warnings.append("Relative weakness is appearing on elevated volume.")
    elif trend == "Weakening":
        status = "Warning"
        adjustment = -5
        confidence = "Medium"
        warnings.append("Ticker is underperforming its comparison benchmark.")
    elif trend == "Improving" and volume == "Not confirmed":
        status = "Warning"
        adjustment = 0
        confidence = "Medium"
        warnings.append("Relative strength is not confirmed by volume.")
    elif corr == "Weakening":
        status = "Warning"
        adjustment = -5
        confidence = "Low"
        warnings.append("Comparison relationship is weakening.")

    if extension == "Strongly extended":
        warnings.append("Relative move is strongly extended; avoid chasing.")
        adjustment = min(adjustment, 5)
        if status == "Supportive":
            status = "Warning"
    elif extension == "Deep weakness":
        warnings.append("Relative performance is deeply weak versus the benchmark.")
        adjustment = min(adjustment, -5)
        status = "Warning" if status == "Neutral" else status

    adjustment = int(max(RELATIVE_CONTEXT_CONFIG["min_score_adjustment"], min(RELATIVE_CONTEXT_CONFIG["max_score_adjustment"], adjustment)))
    return {
        "relationship_status": status,
        "score_adjustment": adjustment,
        "confidence": confidence,
        "warnings": warnings,
    }


def build_relative_context_summary(metrics: dict) -> str:
    ticker = metrics.get("ticker", "Ticker")
    benchmark = metrics.get("benchmark", "benchmark")
    status = metrics.get("relationship_status", "Unavailable")
    if not metrics.get("available"):
        return metrics.get("interpretation", "Relative context unavailable.")

    rel20 = _format_pct(metrics.get("relative_20d"))
    extension = metrics.get("relative_extension_label", "Unavailable").lower()
    corr = metrics.get("correlation_trend", "Unavailable").lower()
    beta = metrics.get("beta_trend", "Unavailable").lower()
    volume = metrics.get("volume_confirmation", "Unavailable").lower()

    if status == "Supportive":
        return (
            f"{ticker} is outperforming {benchmark} over the last 20 trading days ({rel20}). "
            f"Volume is {volume}, the relationship looks {corr}, and beta is {beta}. "
            f"The move is {extension}, so relative context supports the setup without becoming the main signal."
        )
    if status == "Warning":
        return (
            f"{ticker} relative context needs caution versus {benchmark}. "
            f"20D relative performance is {rel20}, volume is {volume}, and the move is {extension}."
        )
    if status == "Broken":
        return (
            f"{ticker} is no longer moving consistently with {benchmark}. "
            "That does not automatically mean bearish, but comparison confirmation is unreliable."
        )
    if status == "Neutral":
        return (
            f"{ticker} is roughly in line with {benchmark} over the last 20 trading days ({rel20}). "
            "Relative context does not materially change the MR-1 view."
        )
    return "Relative context unavailable because price or benchmark data is missing."


def analyze_relative_context(
    ticker: str,
    benchmark: str,
    ticker_ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame,
    ticker_volume_metrics: dict | None = None,
    benchmark_volume_metrics: dict | None = None,
) -> dict:
    try:
        ticker_aligned, benchmark_aligned = align_price_data(ticker_ohlcv, benchmark_ohlcv)
        if ticker_aligned.empty or benchmark_aligned.empty or len(ticker_aligned.index) < 25:
            return unavailable_context(ticker, benchmark, "Not enough aligned ticker and benchmark data.")

        relative_ratio = calculate_relative_ratio(ticker_aligned["Close"], benchmark_aligned["Close"])
        if relative_ratio.empty:
            return unavailable_context(ticker, benchmark, "Relative ratio could not be calculated.")

        ticker_returns = ticker_aligned["Close"].pct_change()
        benchmark_returns = benchmark_aligned["Close"].pct_change()
        relative_performance = calculate_relative_performance(relative_ratio)
        correlations = calculate_rolling_correlation(
            ticker_returns=ticker_returns,
            benchmark_returns=benchmark_returns,
            windows=RELATIVE_CONTEXT_CONFIG["correlation_windows"],
        )
        betas = calculate_rolling_beta(
            ticker_returns=ticker_returns,
            benchmark_returns=benchmark_returns,
            windows=RELATIVE_CONTEXT_CONFIG["beta_windows"],
        )
        zscore = calculate_relative_zscore(relative_ratio, window=RELATIVE_CONTEXT_CONFIG["relative_z_window"])

        ticker_price_change = _period_return(ticker_aligned["Close"], 20)
        benchmark_price_change = _period_return(benchmark_aligned["Close"], 20)
        metrics = {
            "available": True,
            "ticker": str(ticker or "").upper(),
            "benchmark": str(benchmark or "").upper(),
            **relative_performance,
            **correlations,
            **betas,
            "relative_trend": classify_relative_trend(
                relative_performance.get("relative_5d"),
                relative_performance.get("relative_20d"),
                relative_performance.get("relative_60d"),
            ),
            "relative_zscore": zscore,
            "relative_extension_label": classify_relative_extension(zscore),
            "correlation_trend": classify_correlation_stability(
                correlations.get("correlation_20d"),
                correlations.get("correlation_60d"),
                correlations.get("correlation_120d"),
            ),
            "beta_trend": classify_beta_trend(betas.get("beta_20d"), betas.get("beta_60d")),
            "volume_confirmation": classify_volume_confirmation(
                ticker_price_change=ticker_price_change,
                benchmark_price_change=benchmark_price_change,
                relative_20d=relative_performance.get("relative_20d"),
                ticker_volume_metrics=ticker_volume_metrics,
                benchmark_volume_metrics=benchmark_volume_metrics,
            ),
            "ticker_rvol_20d": _metric_value(ticker_volume_metrics, "rvol_20d", "RVOL 20D", "Relative Volume 20D"),
            "benchmark_rvol_20d": _metric_value(benchmark_volume_metrics, "rvol_20d", "RVOL 20D", "Relative Volume 20D"),
        }
        status = classify_relationship_status(metrics)
        metrics.update(status)
        metrics["history"] = _history_frame(relative_ratio, ticker_returns, benchmark_returns)
        metrics["interpretation"] = build_relative_context_summary(metrics)
        metrics["debug"] = {
            "aligned_rows": int(len(ticker_aligned.index)),
            "missing_dates": int(len(ticker_ohlcv.index.union(benchmark_ohlcv.index)) - len(ticker_aligned.index)),
            "correlation_windows": RELATIVE_CONTEXT_CONFIG["correlation_windows"],
            "beta_windows": RELATIVE_CONTEXT_CONFIG["beta_windows"],
        }
        return metrics
    except Exception as exc:
        return unavailable_context(ticker, benchmark, f"Relative context unavailable: {_clean_error(exc)}")


def _history_frame(relative_ratio: pd.Series, ticker_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=relative_ratio.index)
    frame["Relative Ratio"] = relative_ratio
    rolling_mean = relative_ratio.rolling(RELATIVE_CONTEXT_CONFIG["relative_z_window"]).mean()
    rolling_std = relative_ratio.rolling(RELATIVE_CONTEXT_CONFIG["relative_z_window"]).std().replace(0, np.nan)
    frame["Relative Z-Score"] = (relative_ratio - rolling_mean) / rolling_std
    for window in [20, 60, 120]:
        frame[f"Correlation {window}D"] = ticker_returns.rolling(window).corr(benchmark_returns)
    frame["Correlation YTD"] = _year_to_date_correlation(ticker_returns, benchmark_returns)
    frame["Correlation 52W"] = ticker_returns.rolling(252).corr(benchmark_returns)
    for window in [20, 60]:
        variance = benchmark_returns.rolling(window).var().replace(0, np.nan)
        frame[f"Beta {window}D"] = ticker_returns.rolling(window).cov(benchmark_returns) / variance
    return frame


def _year_to_date_correlation(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=ticker_returns.index, dtype="float64")
    if ticker_returns.empty or benchmark_returns.empty:
        return output

    latest_date = pd.to_datetime(ticker_returns.index.max())
    year_mask = pd.to_datetime(ticker_returns.index).year == latest_date.year
    ticker_ytd = ticker_returns.loc[year_mask]
    benchmark_ytd = benchmark_returns.loc[year_mask]
    if ticker_ytd.shape[0] < 20:
        return output

    output.loc[ticker_ytd.index] = ticker_ytd.expanding(min_periods=20).corr(benchmark_ytd)
    return output


def _standardize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(column[-1] if column[-1] else column[0]) for column in data.columns]
    columns = {str(column).lower(): column for column in data.columns}
    output = pd.DataFrame(index=pd.to_datetime(data.index))
    for target in ["Open", "High", "Low", "Close", "Volume"]:
        source = columns.get(target.lower())
        if source is not None:
            output[target] = pd.to_numeric(data[source], errors="coerce")
    return output.dropna(how="all")


def _period_return(series: pd.Series, days: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] <= days:
        return None
    base = clean.iloc[-days - 1]
    if base == 0 or pd.isna(base):
        return None
    return _valid_float(clean.iloc[-1] / base - 1)


def _metric_value(metrics: dict | None, *keys: str) -> float | None:
    if not metrics:
        return None
    for key in keys:
        if key in metrics:
            value = _valid_float(metrics.get(key))
            if value is not None:
                return value
    return None


def _format_pct(value) -> str:
    number = _valid_float(value)
    return "N/A" if number is None else f"{number:+.1%}"


def _valid_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or np.isinf(number):
        return None
    return number


def _clean_error(exc: Exception) -> str:
    return (str(exc) or exc.__class__.__name__).replace("\n", " ")[:180]
