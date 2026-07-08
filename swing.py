from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from metadata import AssetContext
from performance import calculate_performance_rows, calculate_returns


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


def build_swing_result(
    ticker: str,
    benchmark: str,
    context: AssetContext,
    market_regime: str,
) -> SwingResult:
    assets = _asset_list(ticker=ticker, benchmark=benchmark, context=context)
    ohlc = _download_ohlc(assets, period="2y")
    warnings: list[str] = []

    if ohlc.empty or ticker not in ohlc.get("Close", pd.DataFrame()).columns:
        return _empty_result(ticker=ticker, warning="Swing data is unavailable for this ticker.")

    close = ohlc["Close"].dropna(how="all")
    frame = _calculate_swing_frame(ohlc=ohlc, ticker=ticker, benchmark=benchmark, context=context)
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
        return _empty_result(ticker=ticker, warning="Not enough swing data to calculate indicators.")

    latest = frame.iloc[-1]
    perf_by_ticker = _performance_by_ticker(close)
    peer_rank, peer_count = _peer_rank(ticker=ticker, peers=context.peers, performance=perf_by_ticker)
    signals = _score_signals(
        ticker=ticker,
        benchmark=benchmark,
        context=context,
        latest=latest,
        performance=perf_by_ticker,
        peer_rank=peer_rank,
        peer_count=peer_count,
        market_regime=market_regime,
    )
    score = int(sum(signal.score for signal in signals if signal.contributes_to_score))
    setup_label = _setup_label(score)
    action = _swing_action(
        score=score,
        setup_label=setup_label,
        market_regime=market_regime,
        latest=latest,
    )
    exposure = _swing_exposure(action=action, latest=latest, market_regime=market_regime)
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

    true_range = pd.concat(
        [
            high[ticker] - low[ticker],
            (high[ticker] - close[ticker].shift(1)).abs(),
            (low[ticker] - close[ticker].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["14D ATR"] = true_range.rolling(14).mean()
    data["ATR %"] = data["14D ATR"] / data["Ticker Close"]
    data["20D Realized Volatility"] = close[ticker].pct_change().rolling(20).std() * np.sqrt(252)
    data["QTD Return"] = _rolling_period_return(close[ticker], period="quarter")
    data["YTD Return"] = _rolling_period_return(close[ticker], period="year")

    return data.dropna(subset=["Ticker Close", "20D SMA", "50D SMA", "200D SMA"])


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
) -> tuple[int | None, int]:
    tickers = [ticker, *peers]
    rows = []
    for asset in tickers:
        returns = performance.get(asset, {})
        values = [returns.get(label) for label in ["1M", "3M", "QTD"]]
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
) -> list[SwingSignal]:
    trend_score = 0
    if latest["Ticker Close"] > latest["20D SMA"]:
        trend_score += 8
    if latest["Ticker Close"] > latest["50D SMA"]:
        trend_score += 8
    if latest["20D SMA"] > latest["50D SMA"]:
        trend_score += 4

    ticker_perf = performance.get(ticker, {})
    benchmark_perf = performance.get(benchmark, {})
    rs_score = 0
    if _beats(ticker_perf, benchmark_perf, "1M"):
        rs_score += 10
    if _beats(ticker_perf, benchmark_perf, "3M"):
        rs_score += 10

    market_score = {"Risk-On": 20, "Neutral": 12, "Defensive": 0}.get(market_regime, 0)

    sector_perf = performance.get(context.sector_etf, {})
    industry_perf = performance.get(context.industry_proxy, {})
    sector_score = 0
    for label, points in [("1M", 6), ("3M", 7), ("QTD", 7)]:
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
    vol_score = 5 if pd.notna(atr_pct) and atr_pct <= 0.04 else 3 if pd.notna(atr_pct) and atr_pct <= 0.07 else 0

    return [
        SwingSignal(
            "20D / 50D Trend",
            _status(trend_score, 20),
            int(trend_score),
            20,
            f"Price {latest['Ticker Close']:.2f}; 20D {latest['20D SMA']:.2f}; 50D {latest['50D SMA']:.2f}",
            "Strong when price is above both 20D and 50D averages and 20D is above 50D.",
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
            "Scores whether the ticker is outperforming the benchmark over 1M and 3M swing windows.",
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
            "Scores sector and industry strength across 1M, 3M, and QTD windows.",
        ),
        SwingSignal(
            "Peer Rank",
            _status(peer_score, 15),
            int(peer_score),
            15,
            peer_value,
            "Ranks the ticker against available peers over 1M, 3M, and QTD performance.",
        ),
        SwingSignal(
            "ATR / Volatility Risk",
            _status(vol_score, 5),
            int(vol_score),
            5,
            _format_pct(atr_pct),
            "Lower ATR as a percent of price means the swing setup is cleaner and easier to size.",
        ),
    ]


def _beats(left: dict[str, float | None], right: dict[str, float | None], label: str) -> bool:
    left_value = left.get(label)
    right_value = right.get(label)
    return left_value is not None and right_value is not None and left_value > right_value


def _positive(values: dict[str, float | None], label: str) -> bool:
    value = values.get(label)
    return value is not None and value > 0


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


def _swing_exposure(action: str, latest: pd.Series, market_regime: str) -> float:
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

    if pd.notna(latest.get("ATR %")) and latest["ATR %"] > 0.07:
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
) -> str:
    trend_phrase = (
        "above its 20-day and 50-day moving averages"
        if latest["Ticker Close"] > latest["20D SMA"] and latest["Ticker Close"] > latest["50D SMA"]
        else "not cleanly above its short-term moving averages"
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
        f"{ticker} has a {setup_label.lower()} with a Swing Score of {score}/100. "
        f"The ticker is {trend_phrase}, is {rs_phrase}, and uses {context.sector_etf} / "
        f"{context.industry_proxy} for sector and industry confirmation. "
        f"The strongest driver is {positive_driver}. Main risk: {main_risk}. {risk_phrase}"
    )


def _empty_result(ticker: str, warning: str) -> SwingResult:
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
    )


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"
