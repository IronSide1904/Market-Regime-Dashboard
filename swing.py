from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from config import DEFAULT_TIMEFRAME_PRESET, get_swing_timeframe_profile
from metadata import AssetContext
from performance import calculate_performance_rows, calculate_returns
from volatility import calculate_atr, calculate_realized_volatility, calculate_swing_volatility_context


@dataclass(frozen=True)
class SwingSignal:
    name: str
    status: str
    score: int
    max_score: int
    current_value: str
    explanation: str
    contributes_to_score: bool = True


@dataclass(frozen=True)
class SwingResult:
    score: int
    setup_label: str
    action: str
    exposure: float
    positive_driver: str
    main_risk: str
    signals: list[SwingSignal]
    performance_table: pd.DataFrame
    swing_frame: pd.DataFrame
    close: pd.DataFrame
    warnings: list[str]
    explanation: str
    score_scope: str
    score_horizon: str
    signal_summary: str
    volatility_context: dict


def build_swing_result(
    ticker: str,
    benchmark: str,
    context: AssetContext,
    market_regime: str,
    swing_timeframe: str = "1M",
    volatility_timeframe_config: dict | None = None,
) -> SwingResult:
    profile = get_swing_timeframe_profile(swing_timeframe)
    assets = _asset_list(ticker=ticker, benchmark=benchmark, context=context)
    ohlc = _download_ohlc(assets, period="2y")
    warnings: list[str] = []

    if ohlc.empty or ticker not in ohlc.get("Close", pd.DataFrame()).columns:
        return _empty_result(ticker=ticker, warning="Swing data is unavailable for this ticker.", profile=profile)

    close = ohlc["Close"].dropna(how="all")
    frame = _calculate_swing_frame(
        ohlc=ohlc,
        ticker=ticker,
        benchmark=benchmark,
        context=context,
        volatility_timeframe_config=volatility_timeframe_config,
    )
    performance_assets = [
        ("Selected ticker", ticker, "Ticker"),
        ("Benchmark", benchmark, benchmark),
        ("Sector ETF", context.sector_etf, "Sector"),
        ("Industry proxy", context.industry_proxy, "Industry"),
        ("Theme proxy", context.theme_ticker, "Theme Proxy"),
        ("Sub-industry proxy", context.sub_industry_ticker, "Sub-Industry Proxy"),
    ]
    performance_assets.extend((peer, peer, "Peer") for peer in context.peers)
    performance_table = calculate_performance_rows(close=close, assets=performance_assets)

    if frame.empty:
        return _empty_result(ticker=ticker, warning="Not enough swing data to calculate indicators.", profile=profile)

    ticker_ohlcv = _ticker_ohlcv(ohlc, ticker)
    volatility_context = calculate_swing_volatility_context(ticker_ohlcv, timeframe_config=volatility_timeframe_config)
    latest = frame.iloc[-1]
    perf_by_ticker = _performance_by_ticker(close)
    peer_rank, peer_count = _peer_rank(
        ticker=ticker,
        peers=context.peers,
        performance=perf_by_ticker,
        windows=profile["peer_windows"],
    )
    signals = _score_signals(
        ticker=ticker,
        benchmark=benchmark,
        context=context,
        latest=latest,
        performance=perf_by_ticker,
        peer_rank=peer_rank,
        peer_count=peer_count,
        market_regime=market_regime,
        profile=profile,
    )
    score = int(sum(signal.score for signal in signals if signal.contributes_to_score))
    setup_label = _setup_label(score)
    action = _swing_action(
        score=score,
        setup_label=setup_label,
        market_regime=market_regime,
        latest=latest,
    )
    exposure = _swing_exposure(action=action, latest=latest, market_regime=market_regime, profile=profile)
    positive_driver, main_risk = _drivers(signals)
    explanation = _explanation(
        ticker=ticker,
        benchmark=benchmark,
        context=context,
        score=score,
        setup_label=setup_label,
        latest=latest,
        positive_driver=positive_driver,
        main_risk=main_risk,
        profile=profile,
    )

    return SwingResult(
        score=score,
        setup_label=setup_label,
        action=action,
        exposure=exposure,
        positive_driver=positive_driver,
        main_risk=main_risk,
        signals=signals,
        performance_table=performance_table,
        swing_frame=frame,
        close=close,
        warnings=warnings,
        explanation=explanation,
        score_scope=profile["scope"],
        score_horizon=profile["timeframe"],
        signal_summary=_swing_signal_summary(profile),
        volatility_context=volatility_context,
    )


def _asset_list(ticker: str, benchmark: str, context: AssetContext) -> list[str]:
    assets = [
        ticker,
        benchmark,
        context.sector_etf,
        context.industry_proxy,
        context.theme_ticker,
        context.sub_industry_ticker,
        *context.peers,
    ]
    return sorted({asset for asset in assets if asset})


def _download_ohlc(tickers: list[str], period: str) -> pd.DataFrame:
    downloaded = yf.download(
        tickers,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        fields = [field for field in ["Open", "High", "Low", "Close"] if field in downloaded.columns.get_level_values(0)]
        return downloaded[fields].copy()

    frames = {}
    for field in ["Open", "High", "Low", "Close"]:
        if field in downloaded.columns:
            frames[field] = downloaded[[field]].rename(columns={field: tickers[0]})
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _calculate_swing_frame(
    ohlc: pd.DataFrame,
    ticker: str,
    benchmark: str,
    context: AssetContext,
    volatility_timeframe_config: dict | None = None,
) -> pd.DataFrame:
    close = ohlc["Close"]
    high = ohlc["High"]
    low = ohlc["Low"]

    if ticker not in close.columns:
        return pd.DataFrame()

    data = pd.DataFrame(index=close.index)
    data["Ticker Close"] = close[ticker]
    data["20D SMA"] = close[ticker].rolling(20).mean()
    data["50D SMA"] = close[ticker].rolling(50).mean()
    data["200D SMA"] = close[ticker].rolling(200).mean()
    data["10D Return"] = close[ticker].pct_change(10)
    data["1M Return"] = close[ticker].pct_change(21)
    data["3M Return"] = close[ticker].pct_change(63)

    data["Benchmark Close"] = close[benchmark] if benchmark in close.columns else np.nan
    data["Benchmark 1M Return"] = data["Benchmark Close"].pct_change(21)
    data["Benchmark 3M Return"] = data["Benchmark Close"].pct_change(63)
    data["Ticker/Benchmark"] = data["Ticker Close"] / data["Benchmark Close"]
    data["Ticker/Benchmark 50D SMA"] = data["Ticker/Benchmark"].rolling(50).mean()
    data["Ticker/Benchmark 63D SMA"] = data["Ticker/Benchmark"].rolling(63).mean()

    sector = context.sector_etf
    industry = context.industry_proxy
    data["Sector Close"] = close[sector] if sector in close.columns else np.nan
    data["Industry Close"] = close[industry] if industry in close.columns else np.nan
    data["Ticker/Sector"] = data["Ticker Close"] / data["Sector Close"]
    data["Ticker/Sector 50D SMA"] = data["Ticker/Sector"].rolling(50).mean()
    data["Sector 1M Return"] = data["Sector Close"].pct_change(21)
    data["Sector 3M Return"] = data["Sector Close"].pct_change(63)
    data["Industry 1M Return"] = data["Industry Close"].pct_change(21)
    data["Industry 3M Return"] = data["Industry Close"].pct_change(63)

    volatility_timeframe_config = volatility_timeframe_config or {"atr_window": 14, "realized_vol_window": 20}
    ticker_ohlcv = _ticker_ohlcv(ohlc, ticker)
    atr_window = int(volatility_timeframe_config.get("atr_window", 14))
    realized_vol_window = int(volatility_timeframe_config.get("realized_vol_window", 20))
    data["ATR"] = calculate_atr(ticker_ohlcv, window=atr_window)
    data[f"{atr_window}D ATR"] = data["ATR"]
    data["ATR %"] = data["ATR"] / data["Ticker Close"]
    data["Realized Volatility"] = calculate_realized_volatility(ticker_ohlcv, window=realized_vol_window, annualize=True)
    data[f"{realized_vol_window}D Realized Volatility"] = data["Realized Volatility"]
    data["14D ATR"] = data["ATR"]
    data["20D Realized Volatility"] = data["Realized Volatility"]
    data["QTD Return"] = _rolling_period_return(close[ticker], period="quarter")
    data["YTD Return"] = _rolling_period_return(close[ticker], period="year")

    return data.dropna(subset=["Ticker Close", "20D SMA", "50D SMA", "200D SMA"])


def _ticker_ohlcv(ohlc: pd.DataFrame, ticker: str) -> pd.DataFrame:
    fields = {}
    for field in ["Open", "High", "Low", "Close"]:
        try:
            if field in ohlc and ticker in ohlc[field].columns:
                fields[field] = ohlc[field][ticker]
        except Exception:
            continue
    return pd.DataFrame(fields).dropna(how="all") if fields else pd.DataFrame()


def _rolling_period_return(series: pd.Series, period: str) -> pd.Series:
    returns = []
    for idx in series.index:
        window = series.loc[:idx].dropna()
        if window.empty:
            returns.append(np.nan)
            continue
        current_date = pd.Timestamp(idx)
        if period == "quarter":
            start_month = ((current_date.month - 1) // 3) * 3 + 1
            start_date = pd.Timestamp(current_date.year, start_month, 1)
        else:
            start_date = pd.Timestamp(current_date.year, 1, 1)
        period_window = window[window.index >= start_date]
        returns.append(float(window.iloc[-1] / period_window.iloc[0] - 1) if len(period_window) > 1 else np.nan)
    return pd.Series(returns, index=series.index)


def _performance_by_ticker(close: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        ticker: calculate_returns(close[ticker].dropna())
        for ticker in close.columns
        if not close[ticker].dropna().empty
    }


def _peer_rank(
    ticker: str,
    peers: list[str],
    performance: dict[str, dict[str, float | None]],
    windows: list[str],
) -> tuple[int | None, int]:
    tickers = [ticker, *peers]
    rows = []
    for asset in tickers:
        returns = performance.get(asset, {})
        values = [returns.get(label) for label in windows]
        usable = [value for value in values if value is not None]
        if usable:
            rows.append((asset, sum(usable) / len(usable)))

    if not rows or ticker not in {asset for asset, _ in rows}:
        return None, len(rows)

    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    return [asset for asset, _ in ranked].index(ticker) + 1, len(ranked)


def _score_signals(
    ticker: str,
    benchmark: str,
    context: AssetContext,
    latest: pd.Series,
    performance: dict[str, dict[str, float | None]],
    peer_rank: int | None,
    peer_count: int,
    market_regime: str,
    profile: dict,
) -> list[SwingSignal]:
    trend_score = 0
    for average_column, points in profile["trend_checks"]:
        if pd.notna(latest.get(average_column)) and latest["Ticker Close"] > latest[average_column]:
            trend_score += points
    for short_column, long_column, points in profile["stack_checks"]:
        if pd.notna(latest.get(short_column)) and pd.notna(latest.get(long_column)) and latest[short_column] > latest[long_column]:
            trend_score += points

    ticker_perf = performance.get(ticker, {})
    benchmark_perf = performance.get(benchmark, {})
    rs_score = 0
    for label, points in _weighted_windows(profile["relative_windows"], 20):
        if _beats(ticker_perf, benchmark_perf, label):
            rs_score += points

    market_score = {"Risk-On": 20, "Neutral": 12, "Defensive": 0}.get(market_regime, 0)

    sector_perf = performance.get(context.sector_etf, {})
    industry_perf = performance.get(context.industry_proxy, {})
    sector_score = 0
    for label, points in _weighted_windows(profile["support_windows"], 20):
        if _beats(sector_perf, benchmark_perf, label) or _positive(sector_perf, label):
            sector_score += points / 2
        if _beats(industry_perf, benchmark_perf, label) or _positive(industry_perf, label):
            sector_score += points / 2

    if peer_rank is None or peer_count <= 1:
        peer_score = 8
        peer_value = "Peer data limited"
    else:
        peer_score = max(0, int(round(15 * (peer_count - peer_rank) / max(peer_count - 1, 1))))
        peer_value = f"Rank {peer_rank} of {peer_count}"

    atr_pct = latest.get("ATR %", np.nan)
    vol_score = (
        5
        if pd.notna(atr_pct) and atr_pct <= profile["atr_clean"]
        else 3
        if pd.notna(atr_pct) and atr_pct <= profile["atr_warning"]
        else 0
    )

    return [
        SwingSignal(
            f"{profile['scope']} Trend",
            _status(trend_score, 20),
            int(trend_score),
            20,
            f"Price {latest['Ticker Close']:.2f}; 20D {latest['20D SMA']:.2f}; 50D {latest['50D SMA']:.2f}; 200D {latest['200D SMA']:.2f}",
            f"Trend checks are selected from the {profile['timeframe']} {profile['scope'].lower()} swing horizon.",
        ),
        SwingSignal(
            "1M Momentum",
            _status(_momentum_card_score(ticker_perf, "1M"), 10),
            _momentum_card_score(ticker_perf, "1M"),
            10,
            _format_pct(ticker_perf.get("1M")),
            "Positive 1M return shows core swing momentum is improving.",
            False,
        ),
        SwingSignal(
            "3M Momentum",
            _status(_momentum_card_score(ticker_perf, "3M"), 10),
            _momentum_card_score(ticker_perf, "3M"),
            10,
            _format_pct(ticker_perf.get("3M")),
            "Positive 3M return shows the main swing trend still has support.",
            False,
        ),
        SwingSignal(
            "QTD Strength",
            _status(_momentum_card_score(ticker_perf, "QTD"), 10),
            _momentum_card_score(ticker_perf, "QTD"),
            10,
            _format_pct(ticker_perf.get("QTD")),
            "Positive quarter-to-date performance shows current-quarter leadership.",
            False,
        ),
        SwingSignal(
            "Relative Strength vs Benchmark",
            _status(rs_score, 20),
            int(rs_score),
            20,
            f"{ticker}/{benchmark} ratio {latest['Ticker/Benchmark']:.2f}",
            f"Scores whether the ticker is outperforming the benchmark over {', '.join(profile['relative_windows'])} windows.",
        ),
        SwingSignal(
            "Market Regime",
            _status(market_score, 20),
            int(market_score),
            20,
            market_regime,
            "Market regime from the main MR-1 model should support swing entries.",
        ),
        SwingSignal(
            "Sector / Industry Support",
            _status(sector_score, 20),
            int(round(sector_score)),
            20,
            f"{context.sector_etf} / {context.industry_proxy}",
            f"Scores sector and industry strength across {', '.join(profile['support_windows'])} windows.",
        ),
        SwingSignal(
            "Peer Rank",
            _status(peer_score, 15),
            int(peer_score),
            15,
            peer_value,
            f"Ranks the ticker against available peers over {', '.join(profile['peer_windows'])} performance.",
        ),
        SwingSignal(
            "ATR / Volatility Risk",
            _status(vol_score, 5),
            int(vol_score),
            5,
            _format_pct(atr_pct),
            f"Lower ATR is cleaner. This {profile['scope'].lower()} horizon treats <= {profile['atr_clean']:.1%} as clean.",
        ),
    ]


def _beats(left: dict[str, float | None], right: dict[str, float | None], label: str) -> bool:
    left_value = left.get(label)
    right_value = right.get(label)
    return left_value is not None and right_value is not None and left_value > right_value


def _positive(values: dict[str, float | None], label: str) -> bool:
    value = values.get(label)
    return value is not None and value > 0


def _weighted_windows(windows: list[str], total_points: int) -> list[tuple[str, float]]:
    if not windows:
        return []
    base = total_points / len(windows)
    return [(window, base) for window in windows]


def _momentum_card_score(values: dict[str, float | None], label: str) -> int:
    value = values.get(label)
    if value is None:
        return 0
    if value > 0.05:
        return 10
    if value > 0:
        return 6
    return 0


def _status(score: float, max_score: int) -> str:
    if score >= max_score * 0.75:
        return "Positive"
    if score > 0:
        return "Mixed"
    return "Warning"


def _setup_label(score: int) -> str:
    if score >= 80:
        return "Strong Swing Setup"
    if score >= 65:
        return "Valid Swing Setup"
    if score >= 50:
        return "Watchlist Setup"
    if score >= 35:
        return "Weak Setup"
    return "Bad Setup"


def _swing_action(score: int, setup_label: str, market_regime: str, latest: pd.Series) -> str:
    if latest["Ticker Close"] < latest["200D SMA"]:
        return "HIGH-RISK ONLY"
    if latest["Ticker Close"] < latest["50D SMA"]:
        return "AVOID NEW ENTRY"
    if market_regime == "Defensive":
        return "REDUCE / AVOID"
    if score >= 80 and market_regime == "Risk-On":
        return "BUY / ADD"
    if score >= 65:
        return "BUY SMALL / HOLD"
    if score >= 50:
        return "WATCHLIST"
    return "AVOID"


def _swing_exposure(action: str, latest: pd.Series, market_regime: str, profile: dict | None = None) -> float:
    if action == "BUY / ADD":
        exposure = 1.0
    elif action == "BUY SMALL / HOLD":
        exposure = 0.6
    elif action == "WATCHLIST":
        exposure = 0.3
    elif action == "HIGH-RISK ONLY":
        exposure = 0.2
    else:
        exposure = 0.0

    atr_warning = float((profile or {}).get("atr_warning", 0.07))
    if pd.notna(latest.get("ATR %")) and latest["ATR %"] > atr_warning:
        exposure *= 0.5
    if market_regime == "Neutral":
        exposure = min(exposure, 0.6)
    if market_regime == "Defensive":
        exposure = min(exposure, 0.2)
    return exposure


def _drivers(signals: list[SwingSignal]) -> tuple[str, str]:
    positives = [signal for signal in signals if signal.score > 0]
    risks = [signal for signal in signals if signal.score < signal.max_score * 0.5]
    positive = max(positives, key=lambda signal: signal.score).name if positives else "None"
    risk = min(risks, key=lambda signal: signal.score).name if risks else "No major swing risk"
    return positive, risk


def _explanation(
    ticker: str,
    benchmark: str,
    context: AssetContext,
    score: int,
    setup_label: str,
    latest: pd.Series,
    positive_driver: str,
    main_risk: str,
    profile: dict,
) -> str:
    trend_phrase = (
        "above the selected trend filters"
        if latest["Ticker Close"] > latest["20D SMA"] and latest["Ticker Close"] > latest["50D SMA"]
        else "not cleanly above the selected trend filters"
    )
    rs_phrase = (
        f"outperforming {benchmark}"
        if latest["Ticker/Benchmark"] >= latest["Ticker/Benchmark 50D SMA"]
        else f"not yet outperforming {benchmark}"
    )
    risk_phrase = (
        "ATR is elevated, so position sizing should be controlled."
        if pd.notna(latest.get("ATR %")) and latest["ATR %"] > 0.04
        else "ATR is contained, so the setup is easier to size."
    )
    return (
        f"{ticker} has a {setup_label.lower()} with a {profile['timeframe']} "
        f"{profile['scope'].lower()} Swing Score of {score}/100. "
        f"The ticker is {trend_phrase}, is {rs_phrase}, and uses {context.sector_etf} / "
        f"{context.industry_proxy} for sector and industry confirmation. "
        f"The strongest driver is {positive_driver}. Main risk: {main_risk}. {risk_phrase}"
    )


def _empty_result(ticker: str, warning: str, profile: dict | None = None) -> SwingResult:
    profile = profile or get_swing_timeframe_profile("1M")
    return SwingResult(
        score=0,
        setup_label="Bad Setup",
        action="AVOID",
        exposure=0.0,
        positive_driver="None",
        main_risk=warning,
        signals=[],
        performance_table=pd.DataFrame(),
        swing_frame=pd.DataFrame(),
        close=pd.DataFrame(),
        warnings=[warning],
        explanation=f"{ticker} does not have enough clean data for swing-trading analysis.",
        score_scope=profile["scope"],
        score_horizon=profile["timeframe"],
        signal_summary=_swing_signal_summary(profile),
        volatility_context={
            "available": False,
            "timeframe_preset": DEFAULT_TIMEFRAME_PRESET,
            "volatility_status": "Unavailable",
            "swing_risk_label": "Unavailable",
            "interpretation": warning,
            "warnings": [warning],
        },
    )


def _swing_signal_summary(profile: dict) -> str:
    return (
        f"RS {', '.join(profile['relative_windows'])} | "
        f"Support {', '.join(profile['support_windows'])} | "
        f"Peers {', '.join(profile['peer_windows'])}"
    )


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"
