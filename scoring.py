from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DEFAULT_TIMEFRAME_PRESET, REGIME_RULES, SEARCHABLE_SIGNAL_WEIGHTS
from volume import classify_volume_history


@dataclass(frozen=True)
class SignalResult:
    name: str
    status: str
    score: int
    max_score: int
    current_value: float
    threshold: float
    explanation: str


def get_regime(score: float) -> str:
    if score >= REGIME_RULES["Risk-On"]["min_score"]:
        return "Risk-On"
    if score >= REGIME_RULES["Neutral"]["min_score"]:
        return "Neutral"
    return "Defensive"


def get_exposure(regime: str) -> float:
    return float(REGIME_RULES[regime]["exposure"])


def apply_relative_context_adjustment(base_score: float, relative_context: dict) -> float:
    if not relative_context or not relative_context.get("available"):
        adjustment = 0
    else:
        adjustment = relative_context.get("score_adjustment", 0)
    try:
        adjustment = int(adjustment)
    except (TypeError, ValueError):
        adjustment = 0
    adjustment = max(-10, min(10, adjustment))
    return float(np.clip(float(base_score) + adjustment, 0, 100))


def score_history(
    indicators: pd.DataFrame,
    ticker_ohlcv: pd.DataFrame | None = None,
    shares_float: float | None = None,
    volume_timeframe_config: dict | None = None,
) -> pd.DataFrame:
    scored = indicators.copy()
    weights = SEARCHABLE_SIGNAL_WEIGHTS

    scored["Ticker Trend Score"] = np.where(
        scored["Ticker Close"] > scored["Ticker SMA"], weights["Ticker Trend"], 0
    )
    scored["Benchmark Trend Score"] = np.where(
        scored["Benchmark Close"] > scored["Benchmark SMA"],
        weights["Benchmark Trend"],
        0,
    )
    scored["VIX Regime Score"] = np.where(
        scored["VIX Close"] < scored["VIX SMA"], weights["VIX Regime"], 0
    )
    scored["Relative Strength Score"] = np.where(
        scored["Relative Strength"] >= scored["Relative Strength SMA"],
        weights["Relative Strength"],
        0,
    )

    breadth_points = weights["Market Breadth / Leadership"] / 2
    scored["Breadth Component Score"] = np.where(
        scored["RSP/SPY"] > scored["RSP/SPY SMA"], breadth_points, 0
    )
    scored["Leadership Component Score"] = np.where(
        scored["XLK/XLU"] > scored["XLK/XLU SMA"], breadth_points, 0
    )
    scored["Market Breadth / Leadership Score"] = (
        scored["Breadth Component Score"] + scored["Leadership Component Score"]
    )

    score_columns = [
        "Ticker Trend Score",
        "Benchmark Trend Score",
        "VIX Regime Score",
        "Relative Strength Score",
        "Market Breadth / Leadership Score",
    ]
    scored["MR-1 Core Score"] = scored[score_columns].sum(axis=1).round().astype(int)

    volume_history = _volume_history(
        scored,
        ticker_ohlcv=ticker_ohlcv,
        shares_float=shares_float,
        volume_timeframe_config=volume_timeframe_config,
    )
    if volume_history.empty:
        scored["Volume Context"] = "Unavailable"
        scored["Volume Status"] = "Volume unavailable"
        scored["Volume Adjustment"] = 0
        scored["Volume Timeframe Preset"] = DEFAULT_TIMEFRAME_PRESET
        scored["Volume Short Window"] = np.nan
        scored["Volume Medium Window"] = np.nan
        scored["Volume Long Window"] = np.nan
        scored["Volume Percentile Window"] = np.nan
        scored["RVOL Short"] = np.nan
        scored["RVOL Medium"] = np.nan
        scored["RVOL Long"] = np.nan
        scored["RVOL 20D"] = np.nan
        scored["RVOL 50D"] = np.nan
        scored["Volume Percentile 1Y"] = np.nan
        scored["Volume Z-Score"] = np.nan
        scored["Dollar Volume"] = np.nan
        scored["Daily Float Turnover"] = np.nan
        scored["5D Float Turnover"] = np.nan
        scored["Volume Explanation"] = "Volume context is unavailable because usable ticker volume data was not found."
    else:
        scored = scored.join(volume_history.reindex(scored.index))
        scored["Volume Adjustment"] = scored["Volume Adjustment"].fillna(0).astype(int)
        scored["Volume Context"] = scored["Volume Context"].fillna("Unavailable")
        scored["Volume Status"] = scored["Volume Status"].fillna("Volume unavailable")
        scored["Volume Explanation"] = scored["Volume Explanation"].fillna(
            "Volume context is unavailable because usable ticker volume data was not found."
        )

    scored["MR-1 Score"] = (
        scored["MR-1 Core Score"] + scored["Volume Adjustment"]
    ).clip(lower=0, upper=100).round().astype(int)
    scored["Regime"] = scored["MR-1 Score"].map(get_regime)
    scored["Exposure"] = scored["Regime"].map(get_exposure)

    return scored


def _volume_history(
    scored: pd.DataFrame,
    ticker_ohlcv: pd.DataFrame | None,
    shares_float: float | None,
    volume_timeframe_config: dict | None,
) -> pd.DataFrame:
    if ticker_ohlcv is None or ticker_ohlcv.empty:
        return pd.DataFrame()

    price_df = ticker_ohlcv.copy()
    if "Close" not in price_df.columns:
        price_df["Close"] = scored["Ticker Close"].reindex(price_df.index)
    trend_window = int((volume_timeframe_config or {}).get("trend_window", 50))
    price_df["SMA 50D"] = price_df["Close"].rolling(trend_window).mean()
    price_df["SMA 200D"] = price_df["Close"].rolling(200).mean()

    trend_status = price_df["Close"] > price_df["SMA 200D"]
    vix_status = (scored["VIX Close"] < scored["VIX SMA"]).reindex(price_df.index).fillna(False)
    return classify_volume_history(
        price_df=price_df,
        trend_status=trend_status,
        vix_status=vix_status,
        shares_float=shares_float,
        timeframe_config=volume_timeframe_config,
    )


def apply_volume_context_adjustment(base_score: float, volume_context: dict) -> float:
    try:
        adjustment = int(volume_context.get("volume_adjustment", volume_context.get("adjustment", 0)))
    except (TypeError, ValueError, AttributeError):
        adjustment = 0
    adjustment = max(-15, min(15, adjustment))
    return float(np.clip(float(base_score) + adjustment, 0, 100))


def apply_swing_volatility_adjustment(base_score: float, volatility_context: dict) -> float:
    if not volatility_context or not volatility_context.get("available"):
        return float(base_score)
    status = str(volatility_context.get("volatility_status", "Normal"))
    adjustment = -5 if status == "Extreme" else -2 if status == "Elevated" else 0
    return float(np.clip(float(base_score) + adjustment, 0, 100))


def latest_signal_breakdown(scored: pd.DataFrame, ticker: str, benchmark: str) -> list[SignalResult]:
    latest = scored.iloc[-1]
    weights = SEARCHABLE_SIGNAL_WEIGHTS

    return [
        SignalResult(
            name="Ticker Trend",
            status=_status(latest["Ticker Trend Score"]),
            score=int(latest["Ticker Trend Score"]),
            max_score=weights["Ticker Trend"],
            current_value=float(latest["Ticker Close"]),
            threshold=float(latest["Ticker SMA"]),
            explanation=f"{ticker} is favorable when price is above its trend average.",
        ),
        SignalResult(
            name="Benchmark Trend",
            status=_status(latest["Benchmark Trend Score"]),
            score=int(latest["Benchmark Trend Score"]),
            max_score=weights["Benchmark Trend"],
            current_value=float(latest["Benchmark Close"]),
            threshold=float(latest["Benchmark SMA"]),
            explanation=f"{benchmark} above its trend average means the market backdrop is supportive.",
        ),
        SignalResult(
            name="Volatility",
            status=_status(latest["VIX Regime Score"]),
            score=int(latest["VIX Regime Score"]),
            max_score=weights["VIX Regime"],
            current_value=float(latest["VIX Close"]),
            threshold=float(latest["VIX SMA"]),
            explanation="VIX below its average means volatility pressure is contained.",
        ),
        SignalResult(
            name="Relative Strength",
            status=_status(latest["Relative Strength Score"]),
            score=int(latest["Relative Strength Score"]),
            max_score=weights["Relative Strength"],
            current_value=float(latest["Relative Strength"]),
            threshold=float(latest["Relative Strength SMA"]),
            explanation=f"{ticker}/{benchmark} above its average means the ticker is leading its benchmark.",
        ),
        SignalResult(
            name="Market Breadth / Leadership",
            status=_status(latest["Market Breadth / Leadership Score"]),
            score=int(latest["Market Breadth / Leadership Score"]),
            max_score=weights["Market Breadth / Leadership"],
            current_value=float(latest["RSP/SPY"]),
            threshold=float(latest["RSP/SPY SMA"]),
            explanation="Scores breadth from RSP/SPY and leadership from XLK/XLU.",
        ),
    ]


def main_drivers(signals: list[SignalResult]) -> tuple[str, str]:
    positives = [signal for signal in signals if signal.score > 0]
    negatives = [signal for signal in signals if signal.score == 0]

    positive = max(positives, key=lambda signal: signal.max_score).name if positives else "None"
    negative = max(negatives, key=lambda signal: signal.max_score).name if negatives else "None"
    return positive, negative


def run_backtest(scored: pd.DataFrame) -> pd.DataFrame:
    backtest = scored[["Ticker Close", "Regime", "Exposure"]].copy()
    backtest["Ticker Return"] = backtest["Ticker Close"].pct_change().fillna(0.0)
    backtest["Model Exposure"] = backtest["Exposure"].shift(1).fillna(0.0)
    backtest["Model Return"] = backtest["Model Exposure"] * backtest["Ticker Return"]
    backtest["Buy Hold Equity"] = (1 + backtest["Ticker Return"]).cumprod()
    backtest["Model Equity"] = (1 + backtest["Model Return"]).cumprod()
    return backtest


def backtest_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    ticker_returns = backtest["Ticker Return"]
    model_returns = backtest["Model Return"]

    return {
        "Buy & Hold Total Return": _total_return(backtest["Buy Hold Equity"]),
        "MR-1 Total Return": _total_return(backtest["Model Equity"]),
        "Buy & Hold Max Drawdown": _max_drawdown(backtest["Buy Hold Equity"]),
        "MR-1 Max Drawdown": _max_drawdown(backtest["Model Equity"]),
        "Buy & Hold Volatility": float(ticker_returns.std() * np.sqrt(252)),
        "MR-1 Volatility": float(model_returns.std() * np.sqrt(252)),
        "Time in Market": float(backtest["Model Exposure"].mean()),
    }


def _status(score: float) -> str:
    return "Positive" if score > 0 else "Warning"


def _total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def _max_drawdown(equity: pd.Series) -> float:
    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    return float(drawdown.min())
