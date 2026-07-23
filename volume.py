from __future__ import annotations

import numpy as np
import pandas as pd

from config import CUSTOM_TIMEFRAME_LIMITS, DEFAULT_TIMEFRAME_PRESET, MANUAL_FLOAT_SHARES, TIMEFRAME_PRESETS, VOLUME_CONFIG


UNAVAILABLE_CONTEXT = {
    "context": "Unavailable",
    "adjustment": 0,
    "status": "Volume unavailable",
    "timeframe_preset": DEFAULT_TIMEFRAME_PRESET,
    "volume_short_window": 10,
    "volume_medium_window": 20,
    "volume_long_window": 50,
    "volume_percentile_window": 252,
    "rvol_20d": np.nan,
    "rvol_50d": np.nan,
    "rvol_short": np.nan,
    "rvol_medium": np.nan,
    "rvol_long": np.nan,
    "volume_percentile_1y": np.nan,
    "volume_percentile": np.nan,
    "volume_z_score": np.nan,
    "dollar_volume": np.nan,
    "daily_float_turnover": np.nan,
    "five_day_float_turnover": np.nan,
    "float_turnovers": {},
    "turnover_available": False,
    "turnover_label": "Turnover unavailable",
    "turnover_type": None,
    "turnover_source": None,
    "turnover_denominator": np.nan,
    "avg_daily_turnover": np.nan,
    "five_day_turnover": np.nan,
    "turnover_warning": "No share-count denominator available.",
    "explanation": "Volume context is unavailable because usable ticker volume data was not found.",
}


def resolve_share_count_for_turnover(
    ticker: str,
    finviz_metadata: dict | None = None,
    yfinance_metadata: dict | None = None,
    manual_float_map: dict | None = None,
    latest_price: float | None = None,
) -> dict:
    """
    Resolve the best available denominator for turnover calculations.

    Priority:
    1. Finviz shares_float
    2. yfinance floatShares
    3. manual config shares_float
    4. yfinance sharesOutstanding
    5. marketCap / latest_price estimate
    """
    ticker = str(ticker or "").upper()
    finviz_metadata = finviz_metadata or {}
    yfinance_metadata = yfinance_metadata or finviz_metadata
    manual_float_map = manual_float_map or MANUAL_FLOAT_SHARES

    finviz_float = _number_or_none(finviz_metadata.get("finviz_shares_float"))
    if finviz_float is None and str(finviz_metadata.get("source", "")).lower() == "finviz":
        source_hint = str(finviz_metadata.get("shares_float_source") or "finviz").lower()
        if source_hint == "finviz":
            finviz_float = _number_or_none(finviz_metadata.get("shares_float"))
    if finviz_float:
        return _share_count_result(
            available=True,
            count=finviz_float,
            count_type="shares_float",
            source="finviz",
            is_true_float=True,
        )

    yfinance_float = _number_or_none(
        yfinance_metadata.get("yfinance_float_shares")
        or yfinance_metadata.get("floatShares")
        or (
            yfinance_metadata.get("shares_float")
            if str(yfinance_metadata.get("shares_float_source", "")).lower() == "yfinance_floatshares"
            else None
        )
    )
    if yfinance_float:
        return _share_count_result(
            available=True,
            count=yfinance_float,
            count_type="floatShares",
            source="yfinance",
            is_true_float=True,
        )

    manual_float = _number_or_none(manual_float_map.get(ticker))
    if manual_float:
        return _share_count_result(
            available=True,
            count=manual_float,
            count_type="shares_float",
            source="manual_config",
            is_true_float=True,
        )

    shares_outstanding = _number_or_none(
        yfinance_metadata.get("yfinance_shares_outstanding")
        or yfinance_metadata.get("sharesOutstanding")
        or yfinance_metadata.get("shares_outstanding")
        or finviz_metadata.get("shares_outstanding")
    )
    if shares_outstanding:
        return _share_count_result(
            available=True,
            count=shares_outstanding,
            count_type="sharesOutstanding",
            source="yfinance",
            is_true_float=False,
            warning="Using shares outstanding proxy because shares float is unavailable.",
        )

    market_cap = _number_or_none(
        yfinance_metadata.get("yfinance_market_cap")
        or yfinance_metadata.get("marketCap")
        or yfinance_metadata.get("market_cap")
        or finviz_metadata.get("market_cap")
    )
    price = _number_or_none(latest_price or yfinance_metadata.get("yfinance_price") or yfinance_metadata.get("price") or finviz_metadata.get("price"))
    if market_cap and price:
        estimated = market_cap / price
        if estimated:
            return _share_count_result(
                available=True,
                count=estimated,
                count_type="estimated_shares_outstanding",
                source="estimated",
                is_true_float=False,
                warning="Share turnover proxy estimated from market cap / price. Use as directional only.",
            )

    return _share_count_result(
        available=False,
        count=None,
        count_type=None,
        source=None,
        is_true_float=False,
        warning="No share-count denominator available.",
    )


def calculate_turnover_metrics(
    df: pd.DataFrame,
    share_count_info: dict,
    volume_window: int = 20,
    five_day_window: int = 5,
) -> dict:
    """Calculate turnover metrics using the resolved share-count denominator."""
    share_count_info = share_count_info or {}
    denominator = _number_or_none(share_count_info.get("share_count"))
    if df is None or df.empty or "Volume" not in df.columns or not denominator:
        return {
            "turnover_available": False,
            "turnover_label": "Turnover unavailable",
            "turnover_type": None,
            "denominator": None,
            "denominator_source": None,
            "daily_turnover": None,
            "avg_daily_turnover": None,
            "five_day_turnover": None,
            "warning": share_count_info.get("warning") or "No share-count denominator available.",
        }

    volume = pd.to_numeric(df["Volume"], errors="coerce")
    daily_turnover = _latest_number(volume / denominator)
    avg_daily_turnover = _latest_number(volume.rolling(volume_window, min_periods=max(1, min(volume_window, len(volume)))).mean() / denominator)
    five_day_turnover = _latest_number(volume.rolling(five_day_window, min_periods=max(1, min(five_day_window, len(volume)))).sum() / denominator)
    is_true_float = bool(share_count_info.get("is_true_float"))
    return {
        "turnover_available": daily_turnover is not None,
        "turnover_label": "Float Turnover" if is_true_float else "Share Turnover Proxy",
        "turnover_type": "true_float" if is_true_float else "shares_outstanding_proxy",
        "denominator": denominator,
        "denominator_source": share_count_info.get("share_count_source"),
        "daily_turnover": daily_turnover,
        "avg_daily_turnover": avg_daily_turnover,
        "five_day_turnover": five_day_turnover,
        "warning": share_count_info.get("warning"),
    }


def get_timeframe_config(selected_preset: str | None = None, custom_config: dict | None = None) -> dict:
    preset_name = selected_preset or DEFAULT_TIMEFRAME_PRESET
    base = dict(TIMEFRAME_PRESETS.get(preset_name, TIMEFRAME_PRESETS[DEFAULT_TIMEFRAME_PRESET]))
    base["preset"] = preset_name if preset_name in TIMEFRAME_PRESETS else DEFAULT_TIMEFRAME_PRESET

    if custom_config:
        for key, value in custom_config.items():
            if key not in base:
                continue
            if key == "float_turnover_windows":
                base[key] = sorted({_clamp_int(window, CUSTOM_TIMEFRAME_LIMITS["min_volume_window"], CUSTOM_TIMEFRAME_LIMITS["max_volume_window"]) for window in value})
            elif key in {"atr_window", "realized_vol_window"}:
                min_key = "min_atr_window" if key == "atr_window" else "min_realized_vol_window"
                max_key = "max_atr_window" if key == "atr_window" else "max_realized_vol_window"
                base[key] = _clamp_int(value, CUSTOM_TIMEFRAME_LIMITS[min_key], CUSTOM_TIMEFRAME_LIMITS[max_key])
            elif key.endswith("_window"):
                base[key] = _clamp_int(value, CUSTOM_TIMEFRAME_LIMITS["min_volume_window"], CUSTOM_TIMEFRAME_LIMITS["max_volume_window"])
            else:
                base[key] = value

    return base


def calculate_volume_metrics(
    price_df: pd.DataFrame,
    shares_float: float | None = None,
    timeframe_config: dict | None = None,
    share_count_info: dict | None = None,
) -> pd.DataFrame:
    timeframe_config = timeframe_config or get_timeframe_config()
    data = price_df.copy()
    if "Volume" not in data.columns:
        return pd.DataFrame(index=price_df.index)

    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    short_window = int(timeframe_config["volume_short_window"])
    medium_window = int(timeframe_config["volume_medium_window"])
    long_window = int(timeframe_config["volume_long_window"])
    percentile_window = int(timeframe_config["volume_percentile_window"])

    data[f"Avg Volume {short_window}D"] = data["Volume"].rolling(short_window, min_periods=max(2, min(short_window, len(data)))).mean()
    data[f"Avg Volume {medium_window}D"] = data["Volume"].rolling(medium_window, min_periods=max(2, min(medium_window, len(data)))).mean()
    data[f"Avg Volume {long_window}D"] = data["Volume"].rolling(long_window, min_periods=max(2, min(long_window, len(data)))).mean()
    data["Avg Volume Short"] = data[f"Avg Volume {short_window}D"]
    data["Avg Volume Medium"] = data[f"Avg Volume {medium_window}D"]
    data["Avg Volume Long"] = data[f"Avg Volume {long_window}D"]
    data["Relative Volume Short"] = data["Volume"] / data["Avg Volume Short"]
    data["Relative Volume Medium"] = data["Volume"] / data["Avg Volume Medium"]
    data["Relative Volume Long"] = data["Volume"] / data["Avg Volume Long"]
    data["Volume Percentile"] = data["Volume"].rolling(percentile_window, min_periods=max(2, min(percentile_window, len(data)))).apply(_percentile_rank, raw=False)

    data["Avg Volume 20D"] = data["Avg Volume Medium"]
    data["Avg Volume 50D"] = data["Avg Volume Long"]
    data["Relative Volume 20D"] = data["Relative Volume Medium"]
    data["Relative Volume 50D"] = data["Relative Volume Long"]
    data["Volume Percentile 1Y"] = data["Volume Percentile"]

    volume_std = data["Volume"].rolling(percentile_window, min_periods=max(2, min(percentile_window, len(data)))).std()
    volume_mean = data["Volume"].rolling(percentile_window, min_periods=max(2, min(percentile_window, len(data)))).mean()
    data["Volume Z-Score"] = (data["Volume"] - volume_mean) / volume_std
    data["Dollar Volume"] = data["Close"] * data["Volume"] if "Close" in data.columns else np.nan

    if share_count_info is None and shares_float:
        share_count_info = _share_count_result(
            available=True,
            count=shares_float,
            count_type="shares_float",
            source="legacy",
            is_true_float=True,
        )
    denominator = _number_or_none((share_count_info or {}).get("share_count"))
    turnover_label = (share_count_info or {}).get("turnover_label") or ("Float Turnover" if (share_count_info or {}).get("is_true_float") else "Share Turnover Proxy")
    turnover_type = "true_float" if (share_count_info or {}).get("is_true_float") else "shares_outstanding_proxy" if denominator else None
    denominator_source = (share_count_info or {}).get("share_count_source")
    turnover_warning = (share_count_info or {}).get("warning")

    if denominator:
        data["Daily Turnover"] = data["Volume"] / denominator
        data["Avg Daily Turnover"] = data["Avg Volume Medium"] / denominator
        data["5D Turnover"] = data["Volume"].rolling(5, min_periods=max(1, min(5, len(data)))).sum() / denominator
        for window in timeframe_config.get("float_turnover_windows", []):
            data[f"{window}D Turnover"] = data["Volume"].rolling(window, min_periods=max(1, min(window, len(data)))).sum() / denominator
        data["Daily Float Turnover"] = data["Daily Turnover"] if turnover_type == "true_float" else np.nan
        for window in timeframe_config.get("float_turnover_windows", []):
            data[f"{window}D Float Turnover"] = data[f"{window}D Turnover"] if turnover_type == "true_float" else np.nan
        data["5D Float Turnover"] = data["5D Turnover"] if turnover_type == "true_float" else np.nan
    else:
        data["Daily Turnover"] = np.nan
        data["Avg Daily Turnover"] = np.nan
        data["5D Turnover"] = np.nan
        data["Daily Float Turnover"] = np.nan
        data["5D Float Turnover"] = np.nan
        for window in timeframe_config.get("float_turnover_windows", []):
            data[f"{window}D Turnover"] = np.nan
            data[f"{window}D Float Turnover"] = np.nan
    data["Turnover Available"] = bool(denominator)
    data["Turnover Label"] = turnover_label if denominator else "Turnover unavailable"
    data["Turnover Type"] = turnover_type
    data["Turnover Source"] = denominator_source
    data["Turnover Denominator"] = denominator if denominator else np.nan
    data["Turnover Warning"] = turnover_warning

    return data


def classify_volume_context(
    price_df: pd.DataFrame,
    trend_status: bool,
    vix_status: bool,
    shares_float: float | None = None,
    config: dict | None = None,
    timeframe_config: dict | None = None,
    share_count_info: dict | None = None,
) -> dict:
    settings = config or VOLUME_CONFIG
    timeframe_config = timeframe_config or get_timeframe_config()
    validation_warnings = validate_volume_data(price_df)
    if validation_warnings:
        result = UNAVAILABLE_CONTEXT.copy()
        result["warnings"] = validation_warnings
        return result

    metrics = calculate_volume_metrics(price_df, shares_float=shares_float, timeframe_config=timeframe_config, share_count_info=share_count_info)
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
        timeframe_config=timeframe_config,
        warnings=[],
    )


def calculate_volume_context(
    ohlcv: pd.DataFrame,
    metadata: dict | None = None,
    timeframe_config: dict | None = None,
) -> dict:
    metadata = metadata or {}
    timeframe_config = timeframe_config or get_timeframe_config()
    latest_price = _latest_metric(ohlcv, "Close") if ohlcv is not None and not ohlcv.empty else metadata.get("price")
    share_count_info = resolve_share_count_for_turnover(
        ticker=str(metadata.get("ticker") or ""),
        finviz_metadata=metadata,
        yfinance_metadata=metadata,
        latest_price=latest_price,
    )
    shares_float = share_count_info.get("share_count") if share_count_info.get("is_true_float") else None
    validation_warnings = validate_volume_data(ohlcv)
    warnings = list(validation_warnings)
    if ohlcv is None or ohlcv.empty or "Volume" not in ohlcv.columns:
        result = UNAVAILABLE_CONTEXT.copy()
        result.update(
            {
                "available": False,
                "timeframe_preset": timeframe_config.get("preset", DEFAULT_TIMEFRAME_PRESET),
                "warnings": warnings or ["Volume data is unavailable."],
            }
        )
        return result

    price_df = ohlcv.copy()
    if "Close" in price_df.columns:
        trend_window = int(timeframe_config.get("trend_window", 50))
        price_df["SMA 50D"] = price_df["Close"].rolling(trend_window, min_periods=max(2, min(trend_window, len(price_df)))).mean()
        price_df["SMA 200D"] = price_df["Close"].rolling(200, min_periods=max(2, min(200, len(price_df)))).mean()
    trend_status = bool(price_df["Close"].iloc[-1] > price_df["SMA 200D"].iloc[-1]) if {"Close", "SMA 200D"}.issubset(price_df.columns) else False
    result = classify_volume_context(
        price_df=price_df,
        trend_status=trend_status,
        vix_status=True,
        shares_float=shares_float,
        timeframe_config=timeframe_config,
        share_count_info=share_count_info,
    )
    metrics_for_summary = calculate_volume_metrics(price_df, shares_float, timeframe_config, share_count_info=share_count_info)
    result.update(
        {
            "available": result.get("context") != "Unavailable",
            "current_volume": _latest_metric(ohlcv, "Volume"),
            "avg_volume_short": _latest_metric(metrics_for_summary, "Avg Volume Short"),
            "avg_volume_medium": _latest_metric(metrics_for_summary, "Avg Volume Medium"),
            "avg_volume_long": _latest_metric(metrics_for_summary, "Avg Volume Long"),
            "shares_float": share_count_info.get("share_count") if share_count_info.get("is_true_float") else metadata.get("shares_float"),
            "share_count_info": share_count_info,
            "finviz_average_volume": metadata.get("average_volume"),
            "finviz_current_volume": metadata.get("volume"),
            "finviz_relative_volume": metadata.get("relative_volume"),
            "volume_status": result.get("context"),
            "volume_adjustment": result.get("adjustment", 0),
            "interpretation": result.get("explanation"),
            "warnings": [*warnings, *result.get("warnings", [])],
        }
    )
    for window in timeframe_config.get("float_turnover_windows", []):
        result[f"float_turnover_{window}d"] = result.get("float_turnovers", {}).get(window)
    return result


def classify_volume_history(
    price_df: pd.DataFrame,
    trend_status: pd.Series,
    vix_status: pd.Series,
    shares_float: float | None = None,
    config: dict | None = None,
    timeframe_config: dict | None = None,
    share_count_info: dict | None = None,
) -> pd.DataFrame:
    timeframe_config = timeframe_config or get_timeframe_config()
    metrics = calculate_volume_metrics(price_df, shares_float=shares_float, timeframe_config=timeframe_config, share_count_info=share_count_info)
    if metrics.empty:
        return pd.DataFrame(index=price_df.index)

    previous_close = metrics["Close"].shift(1)
    daily_returns = metrics["Close"] / previous_close - 1
    rows = []
    min_history = min(int(timeframe_config["volume_percentile_window"]), max(2, len(metrics)))
    for position, idx in enumerate(metrics.index):
        if position < min_history - 1:
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
            timeframe_config=timeframe_config,
            warnings=[],
        )
        rows.append(_history_row(result))

    history = pd.DataFrame(rows, index=metrics.index)
    for column in ["Volume", "Avg Volume 20D", "Avg Volume 50D", "Avg Volume Short", "Avg Volume Medium", "Avg Volume Long"]:
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
    timeframe_config: dict,
    warnings: list[str],
) -> dict:
    rvol_short = float(latest.get("Relative Volume Short", np.nan))
    rvol_medium = float(latest.get("Relative Volume Medium", np.nan))
    rvol_long = float(latest.get("Relative Volume Long", np.nan))
    volume_percentile = float(latest.get("Volume Percentile", np.nan))
    close = float(latest["Close"])
    sma_50 = float(latest.get("SMA 50D", np.nan))
    sma_200 = float(latest.get("SMA 200D", np.nan))
    previous_close = float(previous.get("Close", np.nan))
    previous_sma_200 = float(previous.get("SMA 200D", np.nan))

    is_high_volume = rvol_medium > settings["rvol_high"] and volume_percentile > settings["volume_percentile_high"]
    is_extreme_volume = rvol_short > settings["rvol_extreme"] and rvol_medium > settings["rvol_high"] and volume_percentile > settings["volume_percentile_extreme"]
    short_spike_only = rvol_short > settings["rvol_high"] and not rvol_medium > settings["rvol_high"]
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
    elif short_spike_only and daily_return > 0:
        context = "Short-Term Volume Spike"
        adjustment = 5 if above_200 else 0
        status = "Early volume confirmation"
    elif daily_return > 0 and rvol_medium < settings["weak_volume_rvol"]:
        context = "Weak Participation"
        adjustment = -5
        status = "Weak confirmation"

    adjustment = int(max(settings["max_negative_adjustment"], min(settings["max_positive_adjustment"], adjustment)))
    float_turnovers = {
        window: float(latest.get(f"{window}D Float Turnover", np.nan))
        for window in timeframe_config.get("float_turnover_windows", [])
    }
    turnovers = {
        window: float(latest.get(f"{window}D Turnover", np.nan))
        for window in timeframe_config.get("float_turnover_windows", [])
    }
    turnover_label = str(latest.get("Turnover Label") or "Turnover unavailable")
    turnover_type = latest.get("Turnover Type")
    turnover_warning = latest.get("Turnover Warning")
    return {
        "context": context,
        "adjustment": adjustment,
        "status": status,
        "timeframe_preset": timeframe_config.get("preset", DEFAULT_TIMEFRAME_PRESET),
        "volume_short_window": int(timeframe_config["volume_short_window"]),
        "volume_medium_window": int(timeframe_config["volume_medium_window"]),
        "volume_long_window": int(timeframe_config["volume_long_window"]),
        "volume_percentile_window": int(timeframe_config["volume_percentile_window"]),
        "rvol_20d": rvol_medium,
        "rvol_50d": rvol_long,
        "rvol_short": rvol_short,
        "rvol_medium": rvol_medium,
        "rvol_long": rvol_long,
        "volume_percentile_1y": volume_percentile,
        "volume_percentile": volume_percentile,
        "volume_z_score": float(latest.get("Volume Z-Score", np.nan)),
        "dollar_volume": float(latest.get("Dollar Volume", np.nan)),
        "daily_float_turnover": float(latest.get("Daily Float Turnover", np.nan)),
        "five_day_float_turnover": float(latest.get("5D Float Turnover", np.nan)),
        "float_turnovers": float_turnovers,
        "turnover_available": bool(latest.get("Turnover Available", False)),
        "turnover_label": turnover_label,
        "turnover_type": turnover_type,
        "turnover_source": latest.get("Turnover Source"),
        "turnover_denominator": float(latest.get("Turnover Denominator", np.nan)),
        "daily_turnover": float(latest.get("Daily Turnover", np.nan)),
        "avg_daily_turnover": float(latest.get("Avg Daily Turnover", np.nan)),
        "five_day_turnover": float(latest.get("5D Turnover", np.nan)),
        "turnovers": turnovers,
        "turnover_warning": turnover_warning,
        "explanation": _volume_explanation(
            context=context,
            adjustment=adjustment,
            rvol_short=rvol_short,
            rvol_medium=rvol_medium,
            rvol_long=rvol_long,
            volume_percentile=volume_percentile,
            daily_return=daily_return,
            above_200=above_200,
            daily_float_turnover=latest.get("Daily Turnover", np.nan),
            turnover_label=turnover_label,
            timeframe_config=timeframe_config,
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
    if pd.isna(pd.to_numeric(price_df["Volume"], errors="coerce").iloc[-1]):
        warnings.append("Current volume is unavailable.")
    return warnings


def _history_row(result: dict) -> dict:
    return {
        "Volume Context": result["context"],
        "Volume Status": result["status"],
        "Volume Adjustment": result["adjustment"],
        "Volume Timeframe Preset": result.get("timeframe_preset", DEFAULT_TIMEFRAME_PRESET),
        "Volume Short Window": result.get("volume_short_window", 10),
        "Volume Medium Window": result.get("volume_medium_window", 20),
        "Volume Long Window": result.get("volume_long_window", 50),
        "Volume Percentile Window": result.get("volume_percentile_window", 252),
        "RVOL Short": result.get("rvol_short", np.nan),
        "RVOL Medium": result.get("rvol_medium", np.nan),
        "RVOL Long": result.get("rvol_long", np.nan),
        "RVOL 20D": result["rvol_20d"],
        "RVOL 50D": result["rvol_50d"],
        "Volume Percentile 1Y": result["volume_percentile_1y"],
        "Volume Percentile": result.get("volume_percentile", np.nan),
        "Volume Z-Score": result["volume_z_score"],
        "Dollar Volume": result["dollar_volume"],
        "Daily Float Turnover": result["daily_float_turnover"],
        "5D Float Turnover": result["five_day_float_turnover"],
        "Turnover Available": result.get("turnover_available", False),
        "Turnover Label": result.get("turnover_label", "Turnover unavailable"),
        "Turnover Type": result.get("turnover_type"),
        "Turnover Source": result.get("turnover_source"),
        "Turnover Denominator": result.get("turnover_denominator", np.nan),
        "Daily Turnover": result.get("daily_turnover", np.nan),
        "Avg Daily Turnover": result.get("avg_daily_turnover", np.nan),
        "5D Turnover": result.get("five_day_turnover", np.nan),
        "Turnover Warning": result.get("turnover_warning"),
        **{
            f"{window}D Float Turnover": value
            for window, value in result.get("float_turnovers", {}).items()
        },
        **{
            f"{window}D Turnover": value
            for window, value in result.get("turnovers", {}).items()
        },
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
    rvol_short: float,
    rvol_medium: float,
    rvol_long: float,
    volume_percentile: float,
    daily_return: float,
    above_200: bool,
    daily_float_turnover: float,
    turnover_label: str,
    timeframe_config: dict,
) -> str:
    preset = timeframe_config.get("preset", DEFAULT_TIMEFRAME_PRESET)
    short_window = timeframe_config["volume_short_window"]
    medium_window = timeframe_config["volume_medium_window"]
    long_window = timeframe_config["volume_long_window"]
    percentile_window = timeframe_config["volume_percentile_window"]
    rvol_text = "N/A" if pd.isna(rvol_medium) else f"{rvol_medium:.1f}x"
    short_text = "N/A" if pd.isna(rvol_short) else f"{rvol_short:.1f}x"
    long_text = "N/A" if pd.isna(rvol_long) else f"{rvol_long:.1f}x"
    percentile_text = "N/A" if pd.isna(volume_percentile) else f"{volume_percentile:.0f}th percentile"
    percentile_window_text = "1Y" if percentile_window >= 252 else f"{percentile_window}D"
    direction = "rising" if daily_return > 0 else "falling" if daily_return < 0 else "flat"
    trend_text = "above" if above_200 else "below"
    float_text = (
        " Turnover unavailable because no share-count denominator was found."
        if pd.isna(daily_float_turnover)
        else f" Daily volume equals {daily_float_turnover:.1%} of the selected denominator ({turnover_label})."
    )

    if context in {"Accumulation", "Breakout Confirmation"}:
        return (
            f"For the {preset} preset, volume is confirming the move. Current volume is {rvol_text} "
            f"the {medium_window}D average and ranks in the {percentile_text} of the selected "
            f"{percentile_window_text} window. Price is {direction} while "
            f"the ticker is {trend_text} its 200-day average, so volume adds {adjustment:+d} points."
            f"{float_text}"
        )
    if context in {"Distribution", "Panic / Liquidation"}:
        return (
            f"For the {preset} preset, volume is a warning. Current volume is {rvol_text} normal "
            f"{medium_window}D activity and ranks in the {percentile_text} of the selected "
            f"{percentile_window_text} window while price is {direction}. "
            f"That reduces the final MR-1 score by {abs(adjustment)} points.{float_text}"
        )
    if context == "Short-Term Volume Spike":
        return (
            f"For the {preset} preset, volume is spiking versus the {short_window}D average ({short_text}), "
            f"but confirmation is not yet broad versus the {medium_window}D average ({rvol_text}). "
            f"The score only gets a small {adjustment:+d} point adjustment.{float_text}"
        )
    if context == "Weak Participation":
        return (
            f"Price is rising, but volume is only {rvol_text} the {medium_window}D average. "
            f"That weak participation reduces the final MR-1 score by {abs(adjustment)} points.{float_text}"
        )
    if context == "Unavailable":
        return UNAVAILABLE_CONTEXT["explanation"]
    return (
        f"Volume is not sending a strong signal today. For the {preset} preset, trading activity is "
        f"{short_text} / {rvol_text} / {long_text} versus the {short_window}D / {medium_window}D / "
        f"{long_window}D averages and ranks in the {percentile_text} of the selected "
        f"{percentile_window_text} window, so the score adjustment is 0."
        f"{float_text}"
    )


def _clamp_int(value, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = lower
    return max(lower, min(upper, number))


def _latest_metric(frame: pd.DataFrame, column: str):
    if frame is None or frame.empty or column not in frame.columns:
        return None
    clean = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if clean.empty else float(clean.iloc[-1])


def _number_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number <= 0:
        return None
    return number


def _latest_number(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.iloc[-1])


def _share_count_result(
    available: bool,
    count: float | None,
    count_type: str | None,
    source: str | None,
    is_true_float: bool,
    warning: str | None = None,
) -> dict:
    return {
        "share_count_available": available,
        "share_count": count,
        "share_count_type": count_type,
        "share_count_source": source,
        "is_true_float": is_true_float,
        "warning": warning,
    }
