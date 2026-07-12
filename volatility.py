from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_TIMEFRAME_PRESET, SWING_VOLATILITY_CONFIG
from volume import get_timeframe_config


def calculate_atr(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    if ohlcv is None or ohlcv.empty or not {"High", "Low", "Close"}.issubset(ohlcv.columns):
        return pd.Series(dtype="float64")
    high = pd.to_numeric(ohlcv["High"], errors="coerce")
    low = pd.to_numeric(ohlcv["Low"], errors="coerce")
    close = pd.to_numeric(ohlcv["Close"], errors="coerce")
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=max(2, min(window, len(true_range)))).mean()


def calculate_realized_volatility(
    ohlcv: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(ohlcv["Close"], errors="coerce")
    realized = close.pct_change().rolling(window, min_periods=max(2, min(window, len(close)))).std()
    return realized * np.sqrt(252) if annualize else realized


def calculate_swing_volatility_context(
    ohlcv: pd.DataFrame,
    timeframe_config: dict | None = None,
) -> dict:
    config = timeframe_config or get_timeframe_config(DEFAULT_TIMEFRAME_PRESET)
    warnings: list[str] = []
    if ohlcv is None or ohlcv.empty:
        return _unavailable(config, "OHLCV data is unavailable.")
    if not {"High", "Low", "Close"}.issubset(ohlcv.columns):
        return _unavailable(config, "High, Low, or Close data is missing.")

    atr_window = int(config["atr_window"])
    realized_window = int(config["realized_vol_window"])
    if len(ohlcv.index) < max(atr_window, realized_window):
        warnings.append("Not enough history for the full selected volatility window; partial rolling metrics are shown.")

    atr_series = calculate_atr(ohlcv, window=atr_window)
    realized_series = calculate_realized_volatility(
        ohlcv,
        window=realized_window,
        annualize=SWING_VOLATILITY_CONFIG.get("annualize_realized_vol", True),
    )
    close = pd.to_numeric(ohlcv["Close"], errors="coerce")
    latest_close = _last_valid(close)
    latest_atr = _last_valid(atr_series)
    latest_realized = _last_valid(realized_series)
    atr_pct = latest_atr / latest_close if latest_atr is not None and latest_close else None

    if atr_pct is None and latest_realized is None:
        return _unavailable(config, "ATR and realized volatility could not be calculated.", warnings)

    status = _volatility_status(atr_pct, latest_realized)
    risk_label = {
        "Low": "Calm",
        "Normal": "Tradable",
        "Elevated": "Volatile",
        "Extreme": "Dangerous",
    }.get(status, "Tradable")
    stop_distance = atr_pct * 2 if atr_pct is not None else None
    interpretation = _interpretation(
        preset=config.get("preset", DEFAULT_TIMEFRAME_PRESET),
        status=status,
        atr_window=atr_window,
        atr_pct=atr_pct,
        realized_window=realized_window,
        realized_vol=latest_realized,
        stop_distance=stop_distance,
    )

    return {
        "available": True,
        "timeframe_preset": config.get("preset", DEFAULT_TIMEFRAME_PRESET),
        "atr_window": atr_window,
        "atr": latest_atr,
        "atr_pct_price": atr_pct,
        "realized_vol_window": realized_window,
        "realized_vol": latest_realized,
        "realized_vol_annualized": latest_realized if SWING_VOLATILITY_CONFIG.get("annualize_realized_vol", True) else None,
        "volatility_status": status,
        "swing_risk_label": risk_label,
        "suggested_stop_distance": stop_distance,
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _volatility_status(atr_pct: float | None, realized_vol: float | None) -> str:
    atr = atr_pct if atr_pct is not None and not pd.isna(atr_pct) else 0
    realized = realized_vol if realized_vol is not None and not pd.isna(realized_vol) else 0
    if atr >= 0.08 or realized >= 0.80:
        return "Extreme"
    if atr >= 0.05 or realized >= 0.50:
        return "Elevated"
    if atr <= 0.02 and realized <= 0.20:
        return "Low"
    return "Normal"


def _interpretation(
    preset: str,
    status: str,
    atr_window: int,
    atr_pct: float | None,
    realized_window: int,
    realized_vol: float | None,
    stop_distance: float | None,
) -> str:
    atr_text = "N/A" if atr_pct is None or pd.isna(atr_pct) else f"{atr_pct:.1%}"
    realized_text = "N/A" if realized_vol is None or pd.isna(realized_vol) else f"{realized_vol:.1%}"
    stop_text = "N/A" if stop_distance is None or pd.isna(stop_distance) else f"{stop_distance:.1%}"
    if status == "Low":
        read = "Swing risk is calm, though breakouts may need stronger confirmation."
    elif status == "Normal":
        read = "Swing risk is manageable for normal position sizing."
    elif status == "Elevated":
        read = "Use smaller position size or a wider stop; avoid over-sizing."
    else:
        read = "Reduce exposure and require stronger trend and volume confirmation."
    return (
        f"For the {preset} preset, ATR {atr_window}D is {atr_text}, realized volatility "
        f"{realized_window}D is {realized_text}, and a rough 2x ATR stop context is {stop_text}. {read}"
    )


def _unavailable(config: dict, warning: str, warnings: list[str] | None = None) -> dict:
    return {
        "available": False,
        "timeframe_preset": config.get("preset", DEFAULT_TIMEFRAME_PRESET),
        "atr_window": int(config.get("atr_window", 14)),
        "atr": None,
        "atr_pct_price": None,
        "realized_vol_window": int(config.get("realized_vol_window", 20)),
        "realized_vol": None,
        "realized_vol_annualized": None,
        "volatility_status": "Unavailable",
        "swing_risk_label": "Unavailable",
        "suggested_stop_distance": None,
        "interpretation": "Swing volatility is unavailable because usable OHLCV data was not found.",
        "warnings": [*(warnings or []), warning],
    }


def _last_valid(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.iloc[-1])
