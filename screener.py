from __future__ import annotations

import math
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from config import (
    SCREENER_BUCKET_ANALYSIS_CONFIG,
    SCREENER_BUCKET_CHART_OPTIONS,
    SCREENER_CONFIG,
    SCREENER_PEER_MAP,
    SCREENER_PEER_OVERRIDE_CONFIG,
    SCREENER_TARGET_TICKER_CONFIG,
    SCREENER_THEME_GROUPS,
    SCREENER_WATCHLISTS,
    SECTOR_ETF_MAP,
)
from data import normalize_ticker
from finviz_fetcher import configured_finviz_columns, fetch_finviz_export, normalize_finviz_dataframe, remove_dead_local_proxy
from turnover import calculate_turnover_metrics, resolve_share_count_for_turnover


SCREENER_TIMEFRAME_WINDOWS = {
    "5D": 5,
    "10D": 10,
    "1M": 21,
    "2M": 42,
    "3M": 63,
    "4M": 84,
    "6M": 126,
    "8M": 168,
    "10M": 210,
    "YTD": 126,
    "1Y": 252,
}

SCREENER_PERIODS = {
    "5D": "1y",
    "10D": "1y",
    "1M": "1y",
    "2M": "1y",
    "3M": "1y",
    "4M": "2y",
    "6M": "2y",
    "8M": "2y",
    "10M": "2y",
    "YTD": "2y",
    "1Y": "2y",
}

SCREENER_MODES = [
    "Ticker Comparison",
    "Theme Screener",
    "Manual Watchlist",
    "Sector / Industry Screener",
]


def normalize_ticker_list(raw_input: str | list[str]) -> list[str]:
    if isinstance(raw_input, str):
        tokens = re.split(r"[\s,;]+", raw_input)
    else:
        tokens = list(raw_input)

    tickers: list[str] = []
    for token in tokens:
        ticker = normalize_ticker(str(token or ""))
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def parse_peer_override_input(raw_text: str) -> list[str]:
    """
    Parse sidebar peer override input.
    Clean, uppercase, deduplicate, and return valid tickers.
    """
    tickers = normalize_ticker_list(raw_text or "")
    max_tickers = int(SCREENER_PEER_OVERRIDE_CONFIG.get("max_override_tickers", 50))
    return tickers[:max_tickers]


def apply_peer_override_to_buckets(
    buckets: dict,
    override_tickers: list[str],
    apply_to: list[str],
    mode: str = "append",
    target_ticker: str | None = None,
    benchmarks: list[str] | None = None,
) -> dict:
    """
    Apply sidebar peer override to Direct Peers, Sector Peers, and Industry Peers.

    mode='append':
        Add override tickers to existing bucket tickers.

    mode='replace':
        Replace existing bucket tickers with override tickers.

    Always preserve target ticker and benchmarks where required.
    """
    final_buckets = {normalize_ticker(ticker): list(labels) for ticker, labels in (buckets or {}).items() if normalize_ticker(ticker)}
    override = normalize_ticker_list(override_tickers)
    mode = mode if mode in SCREENER_PEER_OVERRIDE_CONFIG.get("allowed_modes", ["append", "replace"]) else "append"
    bucket_label_map = {
        "Direct Peers": "Direct Peer",
        "Sector Peers": "Sector Peer",
        "Industry Peers": "Industry Peer",
    }
    selected_labels = [bucket_label_map[label] for label in apply_to if label in bucket_label_map]

    if mode == "replace" and selected_labels:
        for ticker, labels in list(final_buckets.items()):
            final_buckets[ticker] = [label for label in labels if label not in selected_labels]
            if not final_buckets[ticker]:
                final_buckets.pop(ticker, None)

    for ticker in override:
        if normalize_ticker(ticker) == normalize_ticker(target_ticker or ""):
            continue
        labels = final_buckets.setdefault(ticker, [])
        for label in selected_labels:
            if label not in labels:
                labels.append(label)

    if SCREENER_PEER_OVERRIDE_CONFIG.get("always_include_target", True):
        target = normalize_ticker(target_ticker or "")
        if target:
            labels = final_buckets.setdefault(target, [])
            if "Target" not in labels:
                labels.insert(0, "Target")

    if SCREENER_PEER_OVERRIDE_CONFIG.get("always_include_benchmarks", True):
        clean_benchmarks = normalize_ticker_list(benchmarks or [])
        for index, benchmark in enumerate(clean_benchmarks):
            label = "Benchmark" if index == 0 else "Market Benchmark"
            labels = final_buckets.setdefault(benchmark, [])
            if label not in labels:
                labels.append(label)

    return final_buckets


def build_final_comparison_universe(
    target_ticker: str,
    auto_buckets: dict,
    peer_override_enabled: bool,
    peer_override_tickers: list[str],
    peer_override_mode: str,
    peer_override_apply_to: list[str],
    benchmark: str,
    market_benchmark: str,
) -> dict:
    """
    Build the final screener universe after applying sidebar overrides.
    This final universe must be used by tables, charts, medians, ranks, and overlay calculations.
    """
    base_buckets = {normalize_ticker(ticker): list(labels) for ticker, labels in (auto_buckets or {}).items() if normalize_ticker(ticker)}
    override = normalize_ticker_list(peer_override_tickers)
    metadata = {
        ticker: {
            "bucket_source": "auto_detected",
            "override_applied": False,
            "override_mode": "",
        }
        for ticker in base_buckets
    }

    if peer_override_enabled and override:
        final_buckets = apply_peer_override_to_buckets(
            buckets=base_buckets,
            override_tickers=override,
            apply_to=peer_override_apply_to,
            mode=peer_override_mode,
            target_ticker=target_ticker,
            benchmarks=[benchmark, market_benchmark],
        )
        override_set = set(override)
        for ticker in final_buckets:
            was_auto = ticker in base_buckets
            used_override = ticker in override_set
            metadata[ticker] = {
                "bucket_source": "auto_detected + sidebar_override" if was_auto and used_override else "sidebar_override" if used_override else "auto_detected",
                "override_applied": bool(used_override),
                "override_mode": peer_override_mode if used_override else "",
            }
    else:
        final_buckets = apply_peer_override_to_buckets(
            buckets=base_buckets,
            override_tickers=[],
            apply_to=[],
            mode="append",
            target_ticker=target_ticker,
            benchmarks=[benchmark, market_benchmark],
        )
        for ticker in final_buckets:
            metadata.setdefault(
                ticker,
                {
                    "bucket_source": "auto_detected",
                    "override_applied": False,
                    "override_mode": "",
                },
            )

    tickers = normalize_ticker_list(list(final_buckets.keys()))
    return {
        "tickers": tickers,
        "buckets": final_buckets,
        "bucket_metadata": metadata,
        "override_enabled": bool(peer_override_enabled and override),
        "override_tickers": override,
        "override_mode": peer_override_mode,
        "override_apply_to": peer_override_apply_to,
    }


def build_ticker_comparison_universe(
    target_ticker: str,
    peer_map: dict,
    theme_groups: dict,
    metadata: dict | None = None,
    include_benchmarks: bool = True,
) -> dict:
    target = normalize_ticker(target_ticker)
    mapped = peer_map.get(target, {})
    theme = detect_theme_for_ticker(target, theme_groups=theme_groups, peer_map=peer_map, metadata=metadata)
    benchmark = normalize_ticker(mapped.get("benchmark", SCREENER_CONFIG["default_benchmark"]))
    market_benchmark = normalize_ticker(mapped.get("market_benchmark", SCREENER_CONFIG["default_market_benchmark"]))
    sector = metadata.get("sector") if metadata else None
    industry = metadata.get("industry") if metadata else None

    max_auto_peers = int(SCREENER_TARGET_TICKER_CONFIG.get("max_auto_peers", 25))
    direct_peers = normalize_ticker_list(mapped.get("direct_peers", []))[:max_auto_peers]
    theme_peers = normalize_ticker_list(theme_groups.get(theme, [])) if theme else []

    tickers: list[str] = []
    buckets: dict[str, list[str]] = {}

    def add(symbol: str, bucket: str) -> None:
        normalized = normalize_ticker(symbol)
        if not normalized:
            return
        if normalized not in tickers:
            tickers.append(normalized)
        buckets.setdefault(normalized, [])
        if bucket not in buckets[normalized]:
            buckets[normalized].append(bucket)

    if SCREENER_TARGET_TICKER_CONFIG.get("include_target_in_results", True):
        add(target, "Target")
    for peer in direct_peers:
        add(peer, "Direct Peer")
    for peer in theme_peers:
        if peer != target:
            add(peer, "Theme Peer")
    if include_benchmarks:
        add(benchmark, "Benchmark")
        add(market_benchmark, "Market Benchmark")

    return {
        "target": target,
        "theme": theme,
        "sector": sector,
        "industry": industry,
        "benchmark": benchmark,
        "market_benchmark": market_benchmark,
        "tickers": tickers,
        "buckets": buckets,
    }


def detect_theme_for_ticker(
    ticker: str,
    theme_groups: dict,
    peer_map: dict | None = None,
    metadata: dict | None = None,
) -> str | None:
    normalized = normalize_ticker(ticker)
    mapped = (peer_map or {}).get(normalized, {})
    if mapped.get("theme"):
        return str(mapped["theme"])

    for theme, tickers in theme_groups.items():
        if normalized in normalize_ticker_list(tickers):
            return str(theme)

    sector = str((metadata or {}).get("sector") or "")
    industry = str((metadata or {}).get("industry") or "")
    text = f"{sector} {industry}".lower()
    if "semiconductor" in text:
        return "AI Semiconductors"
    if "software" in text:
        return "AI Software"
    if "security" in text:
        return "Cybersecurity"
    return None


def calculate_target_relative_position(
    target_ticker: str,
    screener_df: pd.DataFrame,
    benchmark: str,
    market_benchmark: str,
    theme_name: str | None = None,
) -> dict:
    target = normalize_ticker(target_ticker)
    if screener_df.empty or target not in set(screener_df["Ticker"]):
        return {"available": False, "summary": f"{target} is not available in the screener results."}

    ranked = screener_df.sort_values("Momentum Trend Score", ascending=False).reset_index(drop=True)
    target_row = ranked.loc[ranked["Ticker"] == target].iloc[0]
    rank = int(ranked.index[ranked["Ticker"] == target][0]) + 1
    theme_label = theme_name or "comparison universe"
    leaders = ranked.loc[ranked["Ticker"] != target].head(1)
    leader_text = ""
    if not leaders.empty:
        leader = leaders.iloc[0]
        leader_gap = float(target_row["Timeframe Return"] - leader["Timeframe Return"])
        leader_text = f"{target} is {'outperforming' if leader_gap >= 0 else 'underperforming'} {leader['Ticker']} by {_format_pct(leader_gap)}."

    summary_parts = [
        f"{target} ranks {rank} / {len(ranked)} in {theme_label}.",
        f"{target} is {'outperforming' if target_row['RS vs SPY'] >= 0 else 'underperforming'} {market_benchmark} by {_format_pct(target_row['RS vs SPY'])}.",
        f"{target} is {'outperforming' if target_row['RS vs Benchmark'] >= 0 else 'underperforming'} {benchmark} by {_format_pct(target_row['RS vs Benchmark'])}.",
        f"{target} is {'outperforming' if target_row['RS vs Theme'] >= 0 else 'underperforming'} the theme median by {_format_pct(target_row['RS vs Theme'])}.",
        f"Trend: {target_row['Trend']}. Volume confirmation: {target_row['Volume']}.",
    ]
    if leader_text:
        summary_parts.insert(3, leader_text)

    return {
        "available": True,
        "target": target,
        "rank": rank,
        "universe_size": len(ranked),
        "theme": theme_label,
        "score": float(target_row["Momentum Trend Score"]),
        "label": str(target_row["Label"]),
        "summary": " ".join(summary_parts),
    }


@st.cache_data(ttl=int(SCREENER_CONFIG["cache_ttl_seconds"]), show_spinner=False)
def fetch_screener_data(tickers: list[str], period: str = "2y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    remove_dead_local_proxy()
    clean_tickers = normalize_ticker_list(tickers)
    if not clean_tickers:
        return {}

    try:
        downloaded = yf.download(
            clean_tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {}

    return _split_downloaded_ohlcv(downloaded=downloaded, tickers=clean_tickers)


@st.cache_data(ttl=int(SCREENER_CONFIG["cache_ttl_seconds"]), show_spinner=False)
def fetch_screener_metadata(tickers: list[str]) -> dict[str, dict]:
    clean_tickers = normalize_ticker_list(tickers)
    metadata: dict[str, dict] = {ticker: {"available": False, "ticker": ticker, "source": "finviz"} for ticker in clean_tickers}
    if not clean_tickers:
        return metadata
    try:
        normalized = normalize_finviz_dataframe(fetch_finviz_export(columns=configured_finviz_columns()))
    except Exception as exc:
        error = str(exc)
        for ticker in clean_tickers:
            metadata[ticker]["error"] = error
        return metadata
    if normalized.empty or "ticker" not in normalized.columns:
        return metadata
    normalized = normalized.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    for ticker in clean_tickers:
        rows = normalized.loc[normalized["ticker"] == ticker]
        if rows.empty:
            metadata[ticker]["error"] = "Ticker not found in Finviz export."
            continue
        row = rows.iloc[0].to_dict()
        row.update({"available": True, "source": "finviz", "error": None})
        metadata[ticker] = row
    return metadata


def calculate_screener_features(
    ticker_data: dict[str, pd.DataFrame],
    benchmark_data: dict[str, pd.DataFrame],
    theme_tickers: list[str],
    timeframe: str,
    metadata: dict | None = None,
) -> pd.DataFrame:
    window = SCREENER_TIMEFRAME_WINDOWS.get(timeframe, 21)
    benchmark_symbol = str(benchmark_data.get("benchmark_symbol", SCREENER_CONFIG["default_benchmark"]))
    market_symbol = str(benchmark_data.get("market_symbol", SCREENER_CONFIG["default_market_benchmark"]))
    target_symbol = normalize_ticker(str(benchmark_data.get("target_symbol", "")))
    include_benchmarks = bool(benchmark_data.get("include_benchmarks", False))
    buckets = benchmark_data.get("buckets", {}) or {}
    bucket_metadata = benchmark_data.get("bucket_metadata", {}) or {}
    benchmark_return = _timeframe_return(benchmark_data.get("benchmark"), window)
    market_return = _timeframe_return(benchmark_data.get("market"), window)
    target_return = _timeframe_return(ticker_data.get(target_symbol), window) if target_symbol else np.nan

    rows: list[dict] = []
    for ticker, frame in ticker_data.items():
        if not include_benchmarks and ticker in {benchmark_symbol, market_symbol}:
            continue
        ticker_metadata = metadata.get(ticker, {}) if isinstance(metadata, dict) else {}
        row = _build_feature_row(
            ticker=ticker,
            frame=frame,
            timeframe=timeframe,
            window=window,
            benchmark_return=benchmark_return,
            market_return=market_return,
            ticker_metadata=ticker_metadata,
        )
        if row:
            row["Bucket"] = _bucket_label(buckets.get(ticker, []))
            ticker_metadata = bucket_metadata.get(ticker, {})
            row["Bucket Source"] = ticker_metadata.get("bucket_source", "auto_detected" if ticker in buckets else "universe")
            row["Override Applied"] = bool(ticker_metadata.get("override_applied", False))
            row["Override Mode"] = ticker_metadata.get("override_mode", "")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    features = pd.DataFrame(rows)
    theme_returns = features.loc[features["Ticker"].isin(theme_tickers), "Timeframe Return"]
    theme_median = float(theme_returns.median()) if not theme_returns.dropna().empty else float(features["Timeframe Return"].median())
    features["Theme Median Return"] = theme_median
    features["RS vs Theme"] = features["Timeframe Return"] - theme_median
    features["RS vs Target"] = features["Timeframe Return"] - target_return if math.isfinite(float(target_return)) else np.nan
    features["RS Composite"] = features[["RS vs Benchmark", "RS vs SPY", "RS vs Theme"]].mean(axis=1)
    bucket_medians = _bucket_return_medians(features)
    features["Peer Median Return"] = bucket_medians.get("Direct Peer", np.nan)
    features["Sector Median Return"] = bucket_medians.get("Sector Peer", np.nan)
    features["Industry Median Return"] = bucket_medians.get("Industry Peer", np.nan)
    features["RS vs Peer Median"] = features["Timeframe Return"] - features["Peer Median Return"]
    features["RS vs Sector"] = features["Timeframe Return"] - features["Sector Median Return"]
    features["RS vs Industry"] = features["Timeframe Return"] - features["Industry Median Return"]
    features["Momentum Percentile"] = features["Timeframe Return"].rank(pct=True)
    features["RS Percentile"] = features["RS Composite"].rank(pct=True)
    features["Theme Percentile"] = features["Timeframe Return"].rank(pct=True)
    features["Sector"] = None
    features["Industry"] = None
    features["Company"] = None
    features["Confidence"] = "Medium"
    if "Bucket" not in features.columns:
        features["Bucket"] = "Universe"
    return features


def score_price_trend(row) -> float:
    close = _number(row.get("Price"))
    sma20 = _number(row.get("SMA 20D"))
    sma50 = _number(row.get("SMA 50D"))
    sma200 = _number(row.get("SMA 200D"))
    if close is None or sma200 is None:
        return 0.0
    if all(value is not None for value in (sma20, sma50)) and close > sma20 > sma50 > sma200:
        return 25.0
    if sma50 is not None and close > sma50 and close > sma200:
        return 18.0
    if close > sma200:
        return 10.0
    if sma50 is not None and close > sma50:
        return 5.0
    return 0.0


def score_momentum(row, universe_df: pd.DataFrame) -> float:
    del universe_df
    timeframe_return = _number(row.get("Timeframe Return"))
    percentile = _number(row.get("Momentum Percentile")) or 0.0
    if timeframe_return is None or timeframe_return <= 0:
        return 0.0
    if percentile >= 0.80:
        return 25.0
    if percentile >= 0.50:
        return 18.0
    return 10.0


def score_relative_strength(row, universe_df: pd.DataFrame) -> float:
    del universe_df
    beats_benchmark = (_number(row.get("RS vs Benchmark")) or 0.0) > 0
    beats_market = (_number(row.get("RS vs SPY")) or 0.0) > 0
    beats_theme = (_number(row.get("RS vs Theme")) or 0.0) > 0
    top_relative = (_number(row.get("RS Percentile")) or 0.0) >= 0.70

    if beats_benchmark and beats_market and beats_theme and top_relative:
        return 30.0
    if beats_benchmark and beats_theme:
        return 22.0
    if beats_benchmark:
        return 14.0
    if beats_theme:
        return 7.0
    return 0.0


def score_volume_confirmation(row) -> float:
    timeframe_return = _number(row.get("Timeframe Return")) or 0.0
    rs_vs_benchmark = _number(row.get("RS vs Benchmark")) or 0.0
    rvol = _number(row.get("RVOL 20D"))
    percentile = _number(row.get("Volume Percentile")) or 0.0
    distribution = bool(row.get("Distribution Warning"))

    if distribution:
        return 0.0
    if timeframe_return > 0 and rs_vs_benchmark > 0 and rvol is not None and rvol >= 1.2 and percentile >= 0.70:
        return 10.0
    if rvol is None or rvol >= 0.8:
        return 7.0
    if timeframe_return > 0:
        return 3.0
    return 0.0


def score_risk_volatility(row) -> float:
    atr_pct = _number(row.get("ATR %")) or 0.0
    realized_vol = _number(row.get("Realized Vol 20D")) or 0.0
    dist_20d = abs(_number(row.get("Distance 20D SMA")) or 0.0)
    drawdown = abs(_number(row.get("Recent Drawdown")) or 0.0)

    if atr_pct <= 0.045 and realized_vol <= 0.45 and dist_20d <= 0.10 and drawdown <= 0.12:
        return 10.0
    if atr_pct <= 0.075 and realized_vol <= 0.75 and dist_20d <= 0.18 and drawdown <= 0.22:
        return 6.0
    if atr_pct <= 0.12 and realized_vol <= 1.10:
        return 3.0
    return 0.0


def calculate_momentum_trend_score(features_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty:
        return features_df

    scored = features_df.copy()
    scored["Price Trend Score"] = scored.apply(score_price_trend, axis=1)
    scored["Momentum Score"] = scored.apply(lambda row: score_momentum(row, scored), axis=1)
    scored["Relative Strength Score"] = scored.apply(lambda row: score_relative_strength(row, scored), axis=1)
    scored["Volume Confirmation Score"] = scored.apply(score_volume_confirmation, axis=1)
    scored["Risk Volatility Score"] = scored.apply(score_risk_volatility, axis=1)
    scored["Momentum Trend Score"] = scored[
        [
            "Price Trend Score",
            "Momentum Score",
            "Relative Strength Score",
            "Volume Confirmation Score",
            "Risk Volatility Score",
        ]
    ].sum(axis=1).clip(0, 100)
    scored["Combined Overlay Score"] = scored.apply(_approx_combined_overlay_score, axis=1)
    scored["Label"] = scored["Momentum Trend Score"].apply(_score_label)
    scored["Rank"] = scored["Momentum Trend Score"].rank(method="first", ascending=False).astype(int)
    scored["Trend"] = scored.apply(_trend_status, axis=1)
    scored["Volume"] = scored.apply(_volume_status, axis=1)
    scored["Risk"] = scored.apply(_risk_status, axis=1)
    scored["Warnings"] = scored.apply(_row_warnings, axis=1)
    return scored.sort_values(["Momentum Trend Score", "Timeframe Return"], ascending=False).reset_index(drop=True)


def render_screener_tab() -> None:
    st.subheader("Stock Momentum Screener")
    st.caption(
        "Ticker-driven momentum discovery: compare a target stock against the peers, theme, benchmarks, "
        "and watchlists that actually matter. It does not run full MR-1 analysis for every ticker."
    )

    controls = _render_screener_controls()
    universe_tickers = _resolve_universe(controls)
    max_tickers = int(SCREENER_CONFIG["max_tickers"])
    limited = len(universe_tickers) > max_tickers
    if limited:
        universe_tickers = universe_tickers[:max_tickers]
        st.warning("This universe is large. For performance, the screener is limited to the first 150 tickers.")

    st.markdown(
        f"**Universe:** {len(universe_tickers)} tickers"
        f" | **Benchmark:** {controls['benchmark']}"
        f" | **Market:** {controls['market_benchmark']}"
        f" | **Timeframe:** {controls['timeframe']}"
    )
    if controls.get("mode") == "Ticker Comparison":
        _render_comparison_universe_preview(controls)

    if not universe_tickers:
        st.info("Choose a universe or paste tickers to run the screener.")
        return

    last_run = st.session_state.get("screener_last_run")
    if not controls["run"]:
        if isinstance(last_run, dict) and not last_run.get("sorted_df", pd.DataFrame()).empty:
            st.caption("Showing the last Screener run. Click **Run Screener** to apply any changed form controls.")
            _render_screener_results(
                scored=last_run.get("scored", pd.DataFrame()),
                filtered=last_run.get("filtered", pd.DataFrame()),
                sorted_df=last_run.get("sorted_df", pd.DataFrame()),
                ticker_data=last_run.get("ticker_data", {}),
                skipped=last_run.get("skipped", []),
                controls=last_run.get("controls", controls),
            )
            return

        st.info("Set the controls, then click **Run Screener**.")
        if SCREENER_BUCKET_ANALYSIS_CONFIG.get("enabled", True):
            st.subheader("Bucket Analysis")
            st.info(
                "Run the screener to populate Target, Direct Peers, Sector Peers, Industry Peers, "
                "Theme Peers, Benchmarks, bucket charts, and the All Buckets Overlay."
            )
        return

    period = SCREENER_PERIODS.get(controls["timeframe"], "2y")
    all_symbols = normalize_ticker_list([*universe_tickers, controls["benchmark"], controls["market_benchmark"]])
    with st.spinner(f"Screening {len(universe_tickers)} tickers..."):
        all_data = fetch_screener_data(all_symbols, period=period)
    with st.spinner("Loading Finviz float / turnover metadata..."):
        screener_metadata = fetch_screener_metadata(universe_tickers)

    benchmark = all_data.get(controls["benchmark"])
    market = all_data.get(controls["market_benchmark"])
    if benchmark is None or benchmark.empty:
        st.error(f"Benchmark data is missing for {controls['benchmark']}.")
        return
    if market is None or market.empty:
        st.error(f"Market benchmark data is missing for {controls['market_benchmark']}.")
        return

    ticker_data = {ticker: all_data.get(ticker, pd.DataFrame()) for ticker in universe_tickers}
    ticker_data = {ticker: frame for ticker, frame in ticker_data.items() if frame is not None and not frame.empty}
    skipped = sorted(set(universe_tickers) - set(ticker_data))
    features = calculate_screener_features(
        ticker_data=ticker_data,
        benchmark_data={
            "benchmark": benchmark,
            "market": market,
            "benchmark_symbol": controls["benchmark"],
            "market_symbol": controls["market_benchmark"],
            "target_symbol": controls.get("target_ticker", ""),
            "include_benchmarks": controls.get("include_benchmarks", False),
            "buckets": controls.get("buckets", {}),
            "bucket_metadata": controls.get("bucket_metadata", {}),
        },
        theme_tickers=controls.get("theme_tickers", universe_tickers),
        timeframe=controls["timeframe"],
        metadata=screener_metadata,
    )
    scored = calculate_momentum_trend_score(features)
    filtered = _apply_screener_filters(scored, controls)
    sorted_df = _sort_screener(filtered, controls["sort_by"]).head(int(controls["top_n"]))
    st.session_state["screener_last_run"] = {
        "scored": scored,
        "filtered": filtered,
        "sorted_df": sorted_df,
        "ticker_data": ticker_data,
        "skipped": skipped,
        "controls": controls,
    }
    _render_screener_results(
        scored=scored,
        filtered=filtered,
        sorted_df=sorted_df,
        ticker_data=ticker_data,
        skipped=skipped,
        controls=controls,
    )


def _render_screener_results(
    scored: pd.DataFrame,
    filtered: pd.DataFrame,
    sorted_df: pd.DataFrame,
    ticker_data: dict[str, pd.DataFrame],
    skipped: list[str],
    controls: dict,
) -> None:
    if controls.get("mode") == "Ticker Comparison":
        _render_target_relative_summary(
            target_ticker=controls["target_ticker"],
            scored=scored,
            benchmark=controls["benchmark"],
            market_benchmark=controls["market_benchmark"],
            theme_name=controls.get("detected_theme"),
        )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Screened", f"{len(scored)}")
    col_b.metric("Shown", f"{len(sorted_df)}")
    col_c.metric("Skipped", f"{len(skipped)}")
    if skipped:
        st.caption(f"Skipped missing data: {', '.join(skipped[:20])}{'...' if len(skipped) > 20 else ''}")
    _render_screener_control_impact(
        controls=controls,
        scored_count=len(scored),
        filtered_count=len(filtered),
        shown_count=len(sorted_df),
    )

    if sorted_df.empty:
        st.warning("No tickers passed the selected filters.")
        render_bucket_analysis_section(
            results_df=scored,
            target_ticker=controls.get("target_ticker"),
            ticker_data=ticker_data,
            benchmark=controls["benchmark"],
            market_benchmark=controls["market_benchmark"],
            timeframe=controls["timeframe"],
        )
        return

    display_df = _display_columns(sorted_df, include_target=controls.get("mode") == "Ticker Comparison")
    if controls.get("mode") == "Ticker Comparison" and SCREENER_TARGET_TICKER_CONFIG.get("highlight_target_row", True):
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color: rgba(34, 211, 238, 0.16)" if row.get("Bucket") == "Target" else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
            column_config=_column_config(
                benchmark=controls["benchmark"],
                market_benchmark=controls["market_benchmark"],
            ),
        )
    else:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config(
                benchmark=controls["benchmark"],
                market_benchmark=controls["market_benchmark"],
            ),
        )

    _render_selected_ticker_preview(sorted_df, timeframe=controls["timeframe"], benchmark=controls["benchmark"])

    render_bucket_analysis_section(
        results_df=scored,
        target_ticker=controls.get("target_ticker"),
        ticker_data=ticker_data,
        benchmark=controls["benchmark"],
        market_benchmark=controls["market_benchmark"],
        timeframe=controls["timeframe"],
    )

    with st.expander("Advanced screener columns", expanded=False):
        st.dataframe(
            _advanced_columns(sorted_df),
            use_container_width=True,
            hide_index=True,
            column_config=_advanced_column_config(),
        )


def _render_screener_controls() -> dict:
    default_mode = str(SCREENER_TARGET_TICKER_CONFIG.get("default_mode", "Ticker Comparison"))
    preset = _load_preset_from_session()
    with st.form("screener_controls"):
        st.markdown("**Ticker comparison setup**")
        target_ticker = normalize_ticker(
            st.text_input("Target ticker", value=str(preset.get("target_ticker", "AMD")))
        )
        mode = st.selectbox(
            "Screener mode",
            SCREENER_MODES,
            index=SCREENER_MODES.index(str(preset.get("mode", default_mode))) if str(preset.get("mode", default_mode)) in SCREENER_MODES else 0,
            help="Controls where the ticker universe comes from: target comparison peers, theme list, manual list, or sector ETF list.",
        )
        target_universe = build_ticker_comparison_universe(
            target_ticker=target_ticker,
            peer_map=SCREENER_PEER_MAP,
            theme_groups=SCREENER_THEME_GROUPS,
            include_benchmarks=bool(SCREENER_TARGET_TICKER_CONFIG.get("include_benchmarks_in_results", True)),
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            theme_default = target_universe.get("theme") or str(preset.get("theme", "Mega Cap Tech"))
            theme_options = list(SCREENER_THEME_GROUPS.keys())
            theme = st.selectbox(
                "Theme group",
                theme_options,
                index=theme_options.index(theme_default) if theme_default in theme_options else 0,
                help="Controls theme peers and the median used for RS vs Theme. In Ticker Comparison mode, the detected target theme is preferred.",
            )
            watchlist = st.selectbox(
                "Watchlist",
                list(SCREENER_WATCHLISTS.keys()),
                index=0,
                help="Used only in Theme Screener mode.",
            )
        with col2:
            benchmark = normalize_ticker(
                st.text_input(
                    "Benchmark",
                    value=str(preset.get("benchmark", target_universe.get("benchmark") or SCREENER_CONFIG["default_benchmark"])),
                    help="Primary comparison ticker. Reflected in RS vs Benchmark, benchmark filters, scoring context, and bucket charts.",
                )
            )
            market_benchmark = normalize_ticker(
                st.text_input(
                    "Market benchmark",
                    value=str(preset.get("market_benchmark", target_universe.get("market_benchmark") or SCREENER_CONFIG["default_market_benchmark"])),
                    help="Broad-market comparison ticker. Reflected in the market-relative strength column and score context.",
                )
            )
            timeframe_options = list(SCREENER_TIMEFRAME_WINDOWS.keys())
            timeframe = st.selectbox(
                "Timeframe",
                timeframe_options,
                index=timeframe_options.index(str(preset.get("timeframe", SCREENER_CONFIG["default_timeframe"]))),
                help="Return window for screener momentum, relative strength, ranking, and charts.",
            )
        with col3:
            filters = preset.get("filters", {}) if isinstance(preset.get("filters"), dict) else {}
            min_price = st.number_input(
                "Minimum price",
                min_value=0.0,
                value=float(filters.get("min_price", SCREENER_CONFIG["default_min_price"])),
                step=1.0,
                help="Liquidity/suitability filter. Tickers below this latest price are removed from displayed results.",
            )
            min_dollar_volume = st.number_input(
                "Minimum dollar volume",
                min_value=0.0,
                value=float(filters.get("min_dollar_volume", SCREENER_CONFIG["default_min_dollar_volume"])),
                step=1_000_000.0,
                help="Liquidity filter using latest price times latest volume. Names below this value are removed from displayed results.",
            )
            top_n = st.number_input(
                "Top N",
                min_value=1,
                max_value=int(SCREENER_CONFIG["max_tickers"]),
                value=int(filters.get("top_n", SCREENER_CONFIG["default_top_n"])),
                help="Limits how many filtered rows are shown after sorting.",
            )

        comparison_default = ", ".join(preset.get("comparison_tickers", target_universe["tickers"]))
        comparison_tickers_text = st.text_area(
            "Edit comparison tickers",
            value=comparison_default,
            help="Used in Ticker Comparison mode. Include or remove peers manually; the 150 ticker safety cap still applies.",
        )
        manual_tickers = st.text_area(
            "Manual ticker input",
            value="NVDA, AMD, AVGO, MRVL, MU",
            help="Used in Manual Watchlist mode. Commas, spaces, and new lines are supported.",
        )

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0, step=5)
            only_above_50d = st.checkbox("Only above 50D SMA", value=False)
        with f2:
            only_above_200d = st.checkbox("Only above 200D SMA", value=False)
            only_outperforming_benchmark = st.checkbox("Only outperforming benchmark", value=mode != "Ticker Comparison")
        with f3:
            only_outperforming_theme = st.checkbox("Only outperforming theme", value=False)
            exclude_high_volatility = st.checkbox("Exclude high volatility names", value=False)
        with f4:
            exclude_weak_volume = st.checkbox("Exclude weak volume names", value=False)
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Momentum Trend Score",
                    "Timeframe Return",
                    "RS vs Benchmark",
                    "RS vs Target",
                    "RS vs Theme",
                    "Volume Confirmation Score",
                    "ATR %",
                    "Distance 50D SMA",
                    "Distance 200D SMA",
                ],
            )

        run = st.form_submit_button("Run Screener", type="primary")

    comparison_tickers = normalize_ticker_list(comparison_tickers_text)
    buckets = {ticker: list(values) for ticker, values in (target_universe.get("buckets", {}) or {}).items()}
    peer_override_enabled = bool(
        st.session_state.get(
            "screener_peer_override_enabled",
            SCREENER_PEER_OVERRIDE_CONFIG.get("default_enabled", False),
        )
    )
    peer_override_tickers = parse_peer_override_input(str(st.session_state.get("screener_peer_override_text", "")))
    peer_override_mode = str(
        st.session_state.get(
            "screener_peer_override_mode",
            SCREENER_PEER_OVERRIDE_CONFIG.get("default_mode", "append"),
        )
    ).lower()
    allowed_override_modes = SCREENER_PEER_OVERRIDE_CONFIG.get("allowed_modes", ["append", "replace"])
    if peer_override_mode not in allowed_override_modes:
        peer_override_mode = str(SCREENER_PEER_OVERRIDE_CONFIG.get("default_mode", "append"))
    peer_override_apply_to = [
        value
        for value in st.session_state.get(
            "screener_peer_override_apply_to",
            SCREENER_PEER_OVERRIDE_CONFIG.get("default_apply_to", ["Direct Peers", "Sector Peers", "Industry Peers"]),
        )
        if value in {"Direct Peers", "Sector Peers", "Industry Peers"}
    ]
    if not peer_override_apply_to:
        peer_override_apply_to = list(SCREENER_PEER_OVERRIDE_CONFIG.get("default_apply_to", ["Direct Peers", "Sector Peers", "Industry Peers"]))

    if mode == "Ticker Comparison":
        buckets.setdefault(target_ticker, ["Target"])
        buckets.setdefault(benchmark or str(SCREENER_CONFIG["default_benchmark"]), ["Benchmark"])
        buckets.setdefault(market_benchmark or str(SCREENER_CONFIG["default_market_benchmark"]), ["Market Benchmark"])
        for ticker in comparison_tickers:
            buckets.setdefault(ticker, ["Manual Override"])
        final_universe = build_final_comparison_universe(
            target_ticker=target_ticker,
            auto_buckets=buckets,
            peer_override_enabled=peer_override_enabled,
            peer_override_tickers=peer_override_tickers,
            peer_override_mode=peer_override_mode,
            peer_override_apply_to=peer_override_apply_to,
            benchmark=benchmark or str(SCREENER_CONFIG["default_benchmark"]),
            market_benchmark=market_benchmark or str(SCREENER_CONFIG["default_market_benchmark"]),
        )
        comparison_tickers = final_universe["tickers"]
        buckets = final_universe["buckets"]
        bucket_metadata = final_universe["bucket_metadata"]
    else:
        final_universe = {
            "tickers": comparison_tickers,
            "buckets": buckets,
            "bucket_metadata": {},
            "override_enabled": False,
            "override_tickers": peer_override_tickers,
            "override_mode": peer_override_mode,
            "override_apply_to": peer_override_apply_to,
        }
        bucket_metadata = {}
    if mode == "Sector / Industry Screener":
        st.caption("Sector / Industry Screener currently uses sector ETFs as a lightweight universe. Stock sector/industry peer expansion can be enhanced with Finviz metadata later.")

    preset_payload = _build_preset_payload(
        mode=mode,
        target_ticker=target_ticker,
        theme=theme,
        comparison_tickers=comparison_tickers,
        benchmark=benchmark or str(SCREENER_CONFIG["default_benchmark"]),
        market_benchmark=market_benchmark or str(SCREENER_CONFIG["default_market_benchmark"]),
        timeframe=timeframe,
        filters={"min_price": float(min_price), "min_dollar_volume": float(min_dollar_volume), "top_n": int(top_n)},
        peer_override_enabled=peer_override_enabled,
        peer_override_tickers=peer_override_tickers,
        peer_override_mode=peer_override_mode,
        peer_override_apply_to=peer_override_apply_to,
    )
    with st.expander("Save / load screener preset", expanded=False):
        uploaded = st.file_uploader("Load preset JSON", type=["json"], key="screener_preset_upload")
        if uploaded is not None:
            try:
                st.session_state["screener_loaded_preset"] = json.loads(uploaded.getvalue().decode("utf-8"))
                st.session_state["screener_apply_loaded_preset"] = True
                st.success("Preset loaded. The controls will update on the next rerun.")
            except Exception as exc:
                st.warning(f"Could not load preset: {exc}")
        st.download_button(
            "Download current preset JSON",
            data=json.dumps(preset_payload, indent=2),
            file_name=f"mr1_screener_{target_ticker or 'preset'}.json",
            mime="application/json",
        )

    return {
        "mode": mode,
        "target_ticker": target_ticker,
        "target_universe": target_universe,
        "detected_theme": target_universe.get("theme") or theme,
        "buckets": buckets,
        "bucket_metadata": bucket_metadata,
        "final_universe": final_universe,
        "comparison_tickers": comparison_tickers,
        "include_benchmarks": mode == "Ticker Comparison" and bool(SCREENER_TARGET_TICKER_CONFIG.get("include_benchmarks_in_results", True)),
        "theme_tickers": normalize_ticker_list(SCREENER_THEME_GROUPS.get(target_universe.get("theme") or theme, [])),
        "peer_override_enabled": peer_override_enabled,
        "peer_override_tickers": peer_override_tickers,
        "peer_override_mode": peer_override_mode,
        "peer_override_apply_to": peer_override_apply_to,
        "theme": theme,
        "watchlist": watchlist,
        "manual_tickers": manual_tickers,
        "benchmark": benchmark or str(SCREENER_CONFIG["default_benchmark"]),
        "market_benchmark": market_benchmark or str(SCREENER_CONFIG["default_market_benchmark"]),
        "timeframe": timeframe,
        "min_price": float(min_price),
        "min_dollar_volume": float(min_dollar_volume),
        "top_n": int(top_n),
        "min_score": int(min_score),
        "only_above_50d": bool(only_above_50d),
        "only_above_200d": bool(only_above_200d),
        "only_outperforming_benchmark": bool(only_outperforming_benchmark),
        "only_outperforming_theme": bool(only_outperforming_theme),
        "exclude_high_volatility": bool(exclude_high_volatility),
        "exclude_weak_volume": bool(exclude_weak_volume),
        "sort_by": sort_by,
        "run": bool(run),
    }


def _resolve_universe(controls: dict) -> list[str]:
    mode = controls["mode"]
    if mode == "Ticker Comparison":
        return normalize_ticker_list(controls.get("comparison_tickers", []))
    if mode == "Manual Watchlist":
        return normalize_ticker_list(controls["manual_tickers"])
    if mode == "Sector / Industry Screener":
        return normalize_ticker_list(SECTOR_ETF_MAP.values())
    return normalize_ticker_list(SCREENER_THEME_GROUPS.get(controls["theme"], []))


def _render_comparison_universe_preview(controls: dict) -> None:
    universe = controls.get("target_universe", {})
    tickers = normalize_ticker_list(controls.get("comparison_tickers", []))
    target = controls.get("target_ticker") or universe.get("target")
    override_note = ""
    if controls.get("peer_override_enabled") and controls.get("peer_override_tickers"):
        override_note = (
            f" | Peer override: {controls.get('peer_override_mode', 'append')} "
            f"to {', '.join(controls.get('peer_override_apply_to', []))}"
        )
    st.info(
        f"**{target} comparison universe** | "
        f"Detected theme: {universe.get('theme') or 'Not detected'} | "
        f"Sector: {universe.get('sector') or 'Unavailable'} | "
        f"Industry: {universe.get('industry') or 'Unavailable'} | "
        f"Benchmark: {controls['benchmark']} | Market: {controls['market_benchmark']}"
        f"{override_note}"
    )
    st.caption(f"{target} will be compared against: {', '.join(tickers)}")


def _render_target_relative_summary(
    target_ticker: str,
    scored: pd.DataFrame,
    benchmark: str,
    market_benchmark: str,
    theme_name: str | None,
) -> None:
    summary = calculate_target_relative_position(
        target_ticker=target_ticker,
        screener_df=scored,
        benchmark=benchmark,
        market_benchmark=market_benchmark,
        theme_name=theme_name,
    )
    if not summary.get("available"):
        st.warning(summary.get("summary", "Target summary unavailable."))
        return
    st.markdown(f"### {summary['target']} Relative Position")
    c1, c2, c3 = st.columns(3)
    c1.metric("Universe Rank", f"{summary['rank']} / {summary['universe_size']}")
    c2.metric("Momentum Score", f"{summary['score']:.0f}/100")
    c3.metric("Label", summary["label"])
    st.info(summary["summary"])


def _load_preset_from_session() -> dict:
    preset = st.session_state.get("screener_loaded_preset")
    return preset if isinstance(preset, dict) else {}


def _build_preset_payload(
    mode: str,
    target_ticker: str,
    theme: str,
    comparison_tickers: list[str],
    benchmark: str,
    market_benchmark: str,
    timeframe: str,
    filters: dict,
    peer_override_enabled: bool = False,
    peer_override_tickers: list[str] | None = None,
    peer_override_mode: str = "append",
    peer_override_apply_to: list[str] | None = None,
) -> dict:
    return {
        "mode": mode,
        "target_ticker": target_ticker,
        "theme": theme,
        "comparison_tickers": comparison_tickers,
        "benchmark": benchmark,
        "market_benchmark": market_benchmark,
        "timeframe": timeframe,
        "filters": filters,
        "peer_override_enabled": bool(peer_override_enabled),
        "peer_override_tickers": normalize_ticker_list(peer_override_tickers or []),
        "peer_override_mode": peer_override_mode,
        "peer_override_apply_to": list(peer_override_apply_to or []),
    }


def _bucket_label(buckets: list[str] | tuple[str, ...] | None) -> str:
    values = [str(bucket) for bucket in (buckets or []) if str(bucket)]
    return " / ".join(values) if values else "Universe"


def _bucket_return_medians(features: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    if features.empty or "Bucket" not in features.columns or "Timeframe Return" not in features.columns:
        return medians
    for bucket in ["Direct Peer", "Sector Peer", "Industry Peer"]:
        mask = features["Bucket"].astype(str).str.split(" / ").apply(lambda labels: bucket in labels)
        values = pd.to_numeric(features.loc[mask, "Timeframe Return"], errors="coerce").dropna()
        medians[bucket] = float(values.median()) if not values.empty else np.nan
    return medians


def _build_feature_row(
    ticker: str,
    frame: pd.DataFrame,
    timeframe: str,
    window: int,
    benchmark_return: float,
    market_return: float,
    ticker_metadata: dict | None = None,
) -> dict | None:
    ohlcv = _clean_ohlcv(frame)
    if ohlcv.shape[0] < max(60, min(window + 5, 252)):
        return None

    close = ohlcv["Close"].dropna()
    if close.empty or close.shape[0] <= window:
        return None
    price = float(close.iloc[-1])
    volume = ohlcv["Volume"].dropna()
    current_volume = float(volume.iloc[-1]) if not volume.empty else np.nan
    dollar_volume = price * current_volume if math.isfinite(current_volume) else np.nan
    ret = _timeframe_return(ohlcv, window)
    one_day_return = _period_return(close, 1)
    sma20 = _last_sma(close, 20)
    sma50 = _last_sma(close, 50)
    sma200 = _last_sma(close, 200)
    avg_volume20 = _last_sma(volume, 20)
    avg_volume50 = _last_sma(volume, 50)
    ticker_metadata = ticker_metadata or {}
    share_count_info = resolve_share_count_for_turnover(
        ticker=ticker,
        finviz_metadata=ticker_metadata,
        yfinance_metadata=ticker_metadata,
        latest_price=price,
    )
    turnover_metrics = calculate_turnover_metrics(
        ohlcv,
        share_count_info,
        volume_window=50,
        five_day_window=5,
    )
    denominator = _number(turnover_metrics.get("denominator"))
    is_true_float = bool(share_count_info.get("is_true_float"))
    shares_float = denominator if is_true_float else np.nan
    current_turnover = turnover_metrics.get("daily_turnover")
    average_turnover = turnover_metrics.get("avg_daily_turnover")
    current_float_turnover = current_turnover if is_true_float else np.nan
    average_float_turnover = average_turnover if is_true_float else np.nan
    rvol20 = current_volume / avg_volume20 if avg_volume20 and avg_volume20 > 0 else np.nan
    volume_percentile = _volume_percentile(volume)
    atr_pct = _atr_pct(ohlcv)
    realized_vol = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)) if close.shape[0] >= 21 else np.nan
    recent_high = close.tail(20).max()
    drawdown = (price / recent_high - 1) if recent_high else np.nan

    return {
        "Ticker": ticker,
        "Timeframe": timeframe,
        "Price": price,
        "Timeframe Return": ret,
        "Benchmark Return": benchmark_return,
        "Market Return": market_return,
        "RS vs Benchmark": ret - benchmark_return,
        "RS vs SPY": ret - market_return,
        "SMA 20D": sma20,
        "SMA 50D": sma50,
        "SMA 200D": sma200,
        "Distance 20D SMA": _distance(price, sma20),
        "Distance 50D SMA": _distance(price, sma50),
        "Distance 200D SMA": _distance(price, sma200),
        "Current Volume": current_volume,
        "Avg Volume 20D": avg_volume20,
        "Avg Volume 50D": avg_volume50,
        "Dollar Volume": dollar_volume,
        "Shares Float": shares_float if shares_float is not None else np.nan,
        "Turnover %": current_turnover if current_turnover is not None else np.nan,
        "Average Daily Turnover": average_turnover if average_turnover is not None else np.nan,
        "5D Turnover": turnover_metrics.get("five_day_turnover") if turnover_metrics.get("five_day_turnover") is not None else np.nan,
        "Turnover Type": turnover_metrics.get("turnover_label", "Turnover unavailable"),
        "Turnover Source": turnover_metrics.get("denominator_source") or "Unavailable",
        "Turnover Warning": turnover_metrics.get("warning"),
        "Turnover Denominator": denominator if denominator is not None else np.nan,
        "Current Float Turnover": current_float_turnover,
        "Average Daily Float Turnover": average_float_turnover,
        "RVOL 20D": rvol20,
        "Volume Percentile": volume_percentile,
        "Distribution Warning": bool(one_day_return < -0.02 and (rvol20 if math.isfinite(rvol20) else 0) >= 1.5),
        "ATR %": atr_pct,
        "Realized Vol 20D": realized_vol,
        "Recent Drawdown": drawdown,
    }


def _split_downloaded_ohlcv(downloaded: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if downloaded.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(downloaded.columns, pd.MultiIndex):
        first_level = list(downloaded.columns.get_level_values(0).unique())
        second_level = list(downloaded.columns.get_level_values(1).unique())
        grouped_by_ticker = any(ticker in first_level for ticker in tickers)
        for ticker in tickers:
            columns = {}
            for field in fields:
                series = None
                if grouped_by_ticker and (ticker, field) in downloaded.columns:
                    series = downloaded[(ticker, field)]
                elif not grouped_by_ticker and (field, ticker) in downloaded.columns:
                    series = downloaded[(field, ticker)]
                elif grouped_by_ticker and ticker in first_level and field in second_level:
                    try:
                        series = downloaded[ticker][field]
                    except Exception:
                        series = None
                if series is not None:
                    columns[field] = series
            if columns:
                result[ticker] = pd.DataFrame(columns).dropna(how="all")
    else:
        columns = {field: downloaded[field] for field in fields if field in downloaded.columns}
        if columns and tickers:
            result[tickers[0]] = pd.DataFrame(columns).dropna(how="all")
    return result


def _apply_screener_filters(scored: pd.DataFrame, controls: dict) -> pd.DataFrame:
    if scored.empty:
        return scored
    result = scored.copy()
    result = result[result["Momentum Trend Score"] >= controls["min_score"]]
    result = result[result["Price"] >= controls["min_price"]]
    result = result[result["Dollar Volume"].fillna(0) >= controls["min_dollar_volume"]]
    if controls["only_above_50d"]:
        result = result[result["Distance 50D SMA"] > 0]
    if controls["only_above_200d"]:
        result = result[result["Distance 200D SMA"] > 0]
    if controls["only_outperforming_benchmark"]:
        result = result[result["RS vs Benchmark"] > 0]
    if controls["only_outperforming_theme"]:
        result = result[result["RS vs Theme"] > 0]
    if controls["exclude_high_volatility"]:
        result = result[result["Risk Volatility Score"] >= 6]
    if controls["exclude_weak_volume"]:
        result = result[result["Volume Confirmation Score"] >= 7]
    return result


def _sort_screener(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if df.empty or sort_by not in df.columns:
        return df
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)


def group_screener_results_by_bucket(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if results_df.empty or "Bucket" not in results_df.columns:
        return {}

    bucket_order = [
        "Target",
        "Direct Peer",
        "Theme Peer",
        "Sector Peer",
        "Industry Peer",
        "Benchmark",
        "Market Benchmark",
        "Manual Override",
        "Universe",
    ]
    grouped_frames: dict[str, list[pd.DataFrame]] = {}
    for bucket in bucket_order:
        mask = results_df["Bucket"].astype(str).str.split(" / ").apply(lambda labels: bucket in labels)
        bucket_df = results_df.loc[mask].copy()
        if not bucket_df.empty:
            grouped_frames.setdefault(_bucket_display_name(bucket), []).append(bucket_df)

    grouped: dict[str, pd.DataFrame] = {}
    for bucket_name, frames in grouped_frames.items():
        grouped[bucket_name] = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["Ticker"])
            .sort_values(["Momentum Trend Score", "Timeframe Return"], ascending=False)
            .reset_index(drop=True)
        )
    return grouped


def render_bucket_table(
    bucket_name: str,
    bucket_df: pd.DataFrame,
    benchmark: str = "QQQ",
    market_benchmark: str = "SPY",
) -> None:
    if bucket_df.empty:
        st.info("No tickers in this bucket.")
        return
    st.dataframe(
        _bucket_table_columns(bucket_df),
        use_container_width=True,
        hide_index=True,
        column_config=_bucket_column_config(benchmark=benchmark, market_benchmark=market_benchmark),
    )
    details = _bucket_detail_columns(bucket_df)
    if not details.empty:
        with st.expander(f"{bucket_name} bucket details", expanded=False):
            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True,
                column_config=_bucket_detail_column_config(),
            )


def render_bucket_chart(
    bucket_name: str,
    bucket_df: pd.DataFrame,
    chart_type: str,
    target_ticker: str | None = None,
    target_row: pd.Series | None = None,
) -> None:
    if bucket_df.empty:
        st.info(f"No chart data for {bucket_name}.")
        return

    chart_map = {
        "Combined Overlay Score": "Combined Overlay Score",
        "Momentum Trend Score": "Momentum Trend Score",
        "Average Daily Float Turnover": "Average Daily Turnover",
        "RS vs Target": "RS vs Target",
        "RS vs Benchmark": "RS vs Benchmark",
        "RS vs QQQ": "RS vs Benchmark",
        "RS vs SPY": "RS vs SPY",
        "RS vs Market": "RS vs SPY",
        "RS vs Theme": "RS vs Theme",
        "Timeframe Return": "Timeframe Return",
        "ATR % / Risk": "ATR %",
        "ATR % Price": "ATR %",
    }
    column = chart_map.get(chart_type)
    if not column or column not in bucket_df.columns:
        st.info(f"{chart_type} is unavailable for this bucket.")
        return

    chart_title = _bucket_turnover_chart_title(bucket_df) if chart_type == "Average Daily Float Turnover" else chart_type
    chart_df = bucket_df[["Ticker", column]].copy()
    chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
    chart_df = chart_df.dropna(subset=[column])
    if chart_df.empty:
        st.info(f"{chart_type} unavailable for this bucket.")
        return
    percent_chart = chart_type in {
        "RS vs Target",
        "RS vs Benchmark",
        "RS vs QQQ",
        "RS vs Market",
        "RS vs SPY",
        "RS vs Theme",
        "Timeframe Return",
        "Average Daily Float Turnover",
        "ATR % / Risk",
        "ATR % Price",
    }
    y_values = chart_df[column] * 100 if percent_chart else chart_df[column]
    colors = [
        "#22d3ee" if target_ticker and normalize_ticker(ticker) == normalize_ticker(target_ticker) else "#60a5fa"
        for ticker in chart_df["Ticker"]
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["Ticker"],
                y=y_values,
                marker_color=colors,
                text=[f"{value:.1f}" for value in y_values],
                textposition="outside",
            )
        ]
    )
    if chart_type in {"Combined Overlay Score", "Momentum Trend Score"}:
        fig.add_hline(y=80 if chart_type == "Combined Overlay Score" else 65, line_dash="dash", line_color="#22c55e")
        fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b")
    if target_row is not None and column in target_row.index and target_ticker:
        target_value = _number(target_row.get(column))
        if target_value is not None:
            fig.add_hline(
                y=target_value * 100 if percent_chart else target_value,
                line_dash="dot",
                line_color="#22d3ee",
                annotation_text=f"{normalize_ticker(target_ticker)} reference",
            )
    fig.update_layout(
        title=f"{bucket_name} - {chart_title}",
        height=430,
        margin=dict(l=20, r=20, t=55, b=40),
        template="plotly_dark",
        yaxis_title="%" if percent_chart else "Score",
        xaxis_title="Ticker",
    )
    st.plotly_chart(fig, width="stretch", key=f"bucket_chart_{bucket_name}_{chart_type}")


def _bucket_turnover_chart_title(bucket_df: pd.DataFrame) -> str:
    if bucket_df.empty or "Turnover Type" not in bucket_df.columns:
        return "Average Daily Turnover"
    labels = set(str(value) for value in bucket_df["Turnover Type"].dropna().unique())
    if labels == {"Float Turnover"}:
        return "Average Daily Float Turnover"
    if labels and labels.issubset({"Share Turnover Proxy"}):
        return "Average Daily Share Turnover Proxy"
    return "Average Daily Turnover"


def calculate_overlay_history_for_bucket_tickers(
    tickers: list[str],
    benchmark: str,
    market_benchmark: str,
    timeframe: str,
    max_tickers: int = 10,
    ticker_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    del benchmark, market_benchmark
    selected = normalize_ticker_list(tickers)[:max_tickers]
    if not selected:
        return pd.DataFrame()
    data = ticker_data or fetch_screener_data(selected, period=SCREENER_PERIODS.get(timeframe, "2y"))

    history = pd.DataFrame()
    for ticker in selected:
        frame = _clean_ohlcv(data.get(ticker, pd.DataFrame()))
        if frame.empty or "Close" not in frame.columns:
            continue
        score = _approx_overlay_history(frame=frame, timeframe=timeframe)
        if not score.empty:
            history[ticker] = score
    return history.dropna(how="all")


def render_combined_overlay_history_chart(
    overlay_history_df: pd.DataFrame,
    target_ticker: str | None = None,
    thresholds: dict | None = None,
) -> None:
    if overlay_history_df.empty:
        st.info("Combined overlay history is unavailable for the selected bucket tickers.")
        return

    thresholds = thresholds or {"Full Risk Allowed": 80, "Risk Allowed": 65, "Selective Risk": 50, "Reduce Risk": 35}
    fig = go.Figure()
    for ticker in overlay_history_df.columns:
        is_target = target_ticker and normalize_ticker(ticker) == normalize_ticker(target_ticker)
        fig.add_trace(
            go.Scatter(
                x=overlay_history_df.index,
                y=overlay_history_df[ticker],
                mode="lines",
                name=ticker,
                line=dict(width=4 if is_target else 2),
            )
        )

    if SCREENER_BUCKET_ANALYSIS_CONFIG.get("show_threshold_bands", True):
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(34,197,94,0.16)", line_width=0)
        fig.add_hrect(y0=50, y1=80, fillcolor="rgba(245,158,11,0.13)", line_width=0)
        fig.add_hrect(y0=0, y1=50, fillcolor="rgba(239,68,68,0.15)", line_width=0)
    for label, value in thresholds.items():
        fig.add_hline(y=value, line_dash="dash", line_color="#94a3b8", annotation_text=label)
    fig.update_layout(
        title="All Buckets Overlay - Approximate Combined Score History",
        height=520,
        template="plotly_dark",
        yaxis=dict(title="Combined Overlay Score", range=[0, 100]),
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=95),
    )
    st.plotly_chart(fig, width="stretch", key="screener_all_bucket_overlay_history")


def render_bucket_analysis_section(
    results_df: pd.DataFrame,
    target_ticker: str | None = None,
    ticker_data: dict[str, pd.DataFrame] | None = None,
    benchmark: str = "QQQ",
    market_benchmark: str = "SPY",
    timeframe: str = "1M",
) -> None:
    if not SCREENER_BUCKET_ANALYSIS_CONFIG.get("enabled", True) or results_df.empty:
        return

    grouped = group_screener_results_by_bucket(results_df)
    if not grouped:
        return

    st.subheader("Bucket Analysis")
    st.caption(
        "This is the Screener's cross-sectional decision layer: it shows whether the target is leading "
        "its direct peers, sector peers, industry peers, theme peers, and benchmarks."
    )
    _render_bucket_decision_snapshot(grouped=grouped, results_df=results_df, target_ticker=target_ticker)

    tab_names = [*grouped.keys()]
    if SCREENER_BUCKET_ANALYSIS_CONFIG.get("show_overlay_history_chart", True):
        tab_names.append("All Buckets Overlay")
    tabs = st.tabs(tab_names)

    for tab, bucket_name in zip(tabs, tab_names):
        with tab:
            if bucket_name == "All Buckets Overlay":
                limit = st.slider(
                    "Overlay chart ticker limit",
                    min_value=3,
                    max_value=15,
                    value=int(SCREENER_BUCKET_ANALYSIS_CONFIG.get("overlay_history_max_tickers", 10)),
                    step=1,
                    key="bucket_overlay_limit",
                )
                selected = _select_overlay_history_tickers(results_df, target_ticker=target_ticker, limit=limit)
                st.caption(f"Showing approximate overlay history for: {', '.join(selected)}")
                history = calculate_overlay_history_for_bucket_tickers(
                    tickers=selected,
                    benchmark=benchmark,
                    market_benchmark=market_benchmark,
                    timeframe=timeframe,
                    max_tickers=limit,
                    ticker_data=ticker_data,
                )
                render_combined_overlay_history_chart(history, target_ticker=target_ticker)
                continue

            bucket_df = grouped[bucket_name]
            target_row = _target_row(results_df, target_ticker)
            _render_bucket_summary(bucket_name=bucket_name, bucket_df=bucket_df, target_ticker=target_ticker)
            if SCREENER_BUCKET_ANALYSIS_CONFIG.get("show_bucket_tables", True):
                render_bucket_table(
                    bucket_name=bucket_name,
                    bucket_df=bucket_df,
                    benchmark=benchmark,
                    market_benchmark=market_benchmark,
                )
            if SCREENER_BUCKET_ANALYSIS_CONFIG.get("show_bucket_charts", True):
                default_chart = str(SCREENER_BUCKET_ANALYSIS_CONFIG.get("default_bucket_chart", "Combined Overlay Score"))
                chart_type = st.selectbox(
                    "Chart type",
                    SCREENER_BUCKET_CHART_OPTIONS,
                    index=SCREENER_BUCKET_CHART_OPTIONS.index(default_chart) if default_chart in SCREENER_BUCKET_CHART_OPTIONS else 0,
                    key=f"bucket_chart_type_{bucket_name}",
                )
                render_bucket_chart(
                    bucket_name=bucket_name,
                    bucket_df=bucket_df,
                    chart_type=chart_type,
                    target_ticker=target_ticker,
                    target_row=target_row,
                )


def render_screener_context_snapshot(
    target_ticker: str | None = None,
    title: str = "Latest Screener Bucket Context",
    show_empty: bool = False,
) -> None:
    last_run = st.session_state.get("screener_last_run")
    if not isinstance(last_run, dict):
        if show_empty:
            st.info("No Screener run is available yet. Run the Screener tab to populate peer/bucket context here.")
        return

    scored = last_run.get("scored", pd.DataFrame())
    controls = last_run.get("controls", {})
    if scored is None or scored.empty:
        if show_empty:
            st.info("No Screener run is available yet. Run the Screener tab to populate peer/bucket context here.")
        return

    target = normalize_ticker(target_ticker or controls.get("target_ticker") or "")
    if not target or target not in set(scored["Ticker"]):
        target = normalize_ticker(str(scored.iloc[0].get("Ticker", "")))
    grouped = group_screener_results_by_bucket(scored)

    st.subheader(title)
    st.caption(
        f"From latest Screener run: {controls.get('mode', 'Screener')} | "
        f"TF {controls.get('timeframe', 'N/A')} | Benchmark {controls.get('benchmark', 'N/A')} | "
        f"Market {controls.get('market_benchmark', 'N/A')}"
    )
    _render_bucket_decision_snapshot(grouped=grouped, results_df=scored, target_ticker=target)


def _render_bucket_decision_snapshot(
    grouped: dict[str, pd.DataFrame],
    results_df: pd.DataFrame,
    target_ticker: str | None,
) -> None:
    if results_df.empty:
        return

    target = normalize_ticker(target_ticker or "")
    target_row = _target_row(results_df, target)
    ranked = results_df.sort_values("Momentum Trend Score", ascending=False).reset_index(drop=True)
    if target_row is not None and target in set(ranked["Ticker"]):
        target_rank = int(ranked.index[ranked["Ticker"] == target][0]) + 1
        target_score = _number(target_row.get("Momentum Trend Score")) or 0.0
        target_label = str(target_row.get("Label", "Unavailable"))
    else:
        target_rank = None
        target_score = 0.0
        target_label = "Unavailable"

    summary = _bucket_summary_frame(grouped=grouped, target_ticker=target)
    strongest_bucket = "N/A"
    weakest_bucket = "N/A"
    if not summary.empty:
        strongest_bucket = str(summary.sort_values("Avg Score", ascending=False).iloc[0]["Bucket"])
        weakest_bucket = str(summary.sort_values("Avg Score", ascending=True).iloc[0]["Bucket"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Rank", "N/A" if target_rank is None else f"{target_rank} / {len(ranked)}")
    col2.metric("Target Score", f"{target_score:.0f} / 100")
    col3.metric("Target Label", target_label)
    col4.metric("Strongest Bucket", strongest_bucket)

    if target_row is not None:
        st.info(_bucket_decision_read(target=target, target_row=target_row, summary=summary, weakest_bucket=weakest_bucket))

    if not summary.empty:
        display = summary.copy()
        for column in ["Avg Return", "Target RS"]:
            if column in display.columns:
                display[column] = display[column] * 100
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Score": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=100, format="%.0f"),
                "Avg Return": st.column_config.NumberColumn("Avg Return", format="%.1f%%"),
                "Target RS": st.column_config.NumberColumn("Target RS", format="%.1f%%"),
            },
        )


def _bucket_summary_frame(grouped: dict[str, pd.DataFrame], target_ticker: str | None) -> pd.DataFrame:
    rows = []
    target = normalize_ticker(target_ticker or "")
    for bucket_name, bucket_df in grouped.items():
        if bucket_df.empty:
            continue
        ranked = bucket_df.sort_values("Momentum Trend Score", ascending=False).reset_index(drop=True)
        target_rows = ranked.loc[ranked["Ticker"] == target] if target else pd.DataFrame()
        target_rank = "N/A"
        target_rs = np.nan
        if not target_rows.empty:
            target_rank = f"{int(target_rows.index[0]) + 1} / {len(ranked)}"
            target_return = _number(target_rows.iloc[0].get("Timeframe Return"))
            bucket_return = _number(bucket_df["Timeframe Return"].mean()) if "Timeframe Return" in bucket_df else None
            if target_return is not None and bucket_return is not None:
                target_rs = target_return - bucket_return
        rows.append(
            {
                "Bucket": bucket_name,
                "Tickers": len(bucket_df),
                "Top Ticker": str(ranked.iloc[0].get("Ticker", "N/A")),
                "Avg Score": float(bucket_df["Momentum Trend Score"].mean()) if "Momentum Trend Score" in bucket_df else np.nan,
                "Avg Return": float(bucket_df["Timeframe Return"].mean()) if "Timeframe Return" in bucket_df else np.nan,
                "Target Rank": target_rank,
                "Target RS": target_rs,
            }
        )
    return pd.DataFrame(rows)


def _bucket_decision_read(target: str, target_row: pd.Series, summary: pd.DataFrame, weakest_bucket: str) -> str:
    score = _number(target_row.get("Momentum Trend Score")) or 0.0
    label = str(target_row.get("Label", "Unavailable"))
    rs_target = _format_pct(target_row.get("RS vs Target"))
    rs_peer = _format_pct(target_row.get("RS vs Peer Median"))
    rs_sector = _format_pct(target_row.get("RS vs Sector"))
    rs_industry = _format_pct(target_row.get("RS vs Industry"))
    rank_reads = []
    if not summary.empty:
        for _, row in summary.iterrows():
            target_rank = str(row.get("Target Rank", "N/A"))
            if target_rank != "N/A":
                rank_reads.append(f"{row['Bucket']} {target_rank}")
    rank_text = "; ".join(rank_reads[:4]) if rank_reads else "target is not inside the displayed bucket tabs"
    return (
        f"{target} is {label} at {score:.0f}/100. Bucket rank read: {rank_text}. "
        f"Relative gaps: vs target {rs_target}, vs peer median {rs_peer}, vs sector {rs_sector}, vs industry {rs_industry}. "
        f"Weakest comparison bucket: {weakest_bucket}."
    )


def _render_bucket_summary(bucket_name: str, bucket_df: pd.DataFrame, target_ticker: str | None = None) -> None:
    top = bucket_df.iloc[0] if not bucket_df.empty else {}
    avg_score = float(bucket_df["Momentum Trend Score"].mean()) if "Momentum Trend Score" in bucket_df else 0.0
    target_rank = "N/A"
    if target_ticker and not bucket_df.empty and normalize_ticker(target_ticker) in set(bucket_df["Ticker"]):
        ranked = bucket_df.sort_values("Momentum Trend Score", ascending=False).reset_index(drop=True)
        target_rank = f"{int(ranked.index[ranked['Ticker'] == normalize_ticker(target_ticker)][0]) + 1} / {len(ranked)}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bucket", bucket_name)
    col2.metric("Tickers", f"{len(bucket_df)}")
    col3.metric("Top ticker", str(top.get("Ticker", "N/A")))
    col4.metric("Avg score", f"{avg_score:.0f}")
    if target_ticker:
        st.caption(f"{normalize_ticker(target_ticker)} rank in bucket: {target_rank}")


def _bucket_display_name(bucket: str) -> str:
    if bucket == "Direct Peer":
        return "Direct Peers"
    if bucket == "Theme Peer":
        return "Theme Peers"
    if bucket == "Sector Peer":
        return "Sector Peers"
    if bucket == "Industry Peer":
        return "Industry Peers"
    if bucket in {"Benchmark", "Market Benchmark"}:
        return "Benchmarks"
    return bucket


def _target_row(results_df: pd.DataFrame, target_ticker: str | None) -> pd.Series | None:
    target = normalize_ticker(target_ticker or "")
    if not target or results_df.empty or "Ticker" not in results_df.columns:
        return None
    rows = results_df.loc[results_df["Ticker"] == target]
    return rows.iloc[0] if not rows.empty else None


def _bucket_table_columns(bucket_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Rank",
        "Ticker",
        "Momentum Trend Score",
        "Label",
        "Timeframe Return",
        "RS vs Target",
        "RS vs Benchmark",
        "RS vs SPY",
        "Trend",
        "Volume",
        "Risk",
    ]
    existing = [column for column in columns if column in bucket_df.columns]
    display = bucket_df[existing].copy()
    for column in ["Timeframe Return", "RS vs Target", "RS vs Benchmark", "RS vs SPY", "RS vs Theme"]:
        if column in display.columns:
            display[column] = display[column] * 100
    return display


def _bucket_detail_columns(bucket_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Bucket",
        "Bucket Source",
        "Override Applied",
        "Override Mode",
        "Combined Overlay Score",
        "Shares Float",
        "Turnover %",
        "Average Daily Turnover",
        "5D Turnover",
        "Turnover Type",
        "Turnover Source",
        "Turnover Denominator",
        "Current Float Turnover",
        "Average Daily Float Turnover",
        "RS vs Theme",
        "RS vs Peer Median",
        "RS vs Sector",
        "RS vs Industry",
    ]
    existing = [column for column in columns if column in bucket_df.columns]
    if not existing:
        return pd.DataFrame()
    display = bucket_df[existing].copy()
    for column in [
        "Current Float Turnover",
        "Average Daily Float Turnover",
        "Turnover %",
        "Average Daily Turnover",
        "5D Turnover",
        "RS vs Theme",
        "RS vs Peer Median",
        "RS vs Sector",
        "RS vs Industry",
    ]:
        if column in display.columns:
            display[column] = display[column] * 100
    return display


def _bucket_column_config(benchmark: str = "QQQ", market_benchmark: str = "SPY") -> dict:
    return {
        "Momentum Trend Score": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
        "Timeframe Return": st.column_config.NumberColumn("Timeframe Return", format="%.1f%%"),
        "RS vs Target": st.column_config.NumberColumn("RS vs Target", format="%.1f%%"),
        "RS vs Benchmark": st.column_config.NumberColumn(f"RS vs {benchmark}", format="%.1f%%"),
        "RS vs SPY": st.column_config.NumberColumn(f"RS vs {market_benchmark}", format="%.1f%%"),
    }


def _bucket_detail_column_config() -> dict:
    return {
        "Override Applied": st.column_config.CheckboxColumn("Override Applied"),
        "Combined Overlay Score": st.column_config.ProgressColumn("Overlay", min_value=0, max_value=100, format="%.0f"),
        "Shares Float": st.column_config.NumberColumn("Shares Float", format="%.0f"),
        "Turnover %": st.column_config.NumberColumn("Turnover %", format="%.1f%%"),
        "Average Daily Turnover": st.column_config.NumberColumn("Avg Daily Turnover", format="%.1f%%"),
        "5D Turnover": st.column_config.NumberColumn("5D Turnover", format="%.1f%%"),
        "Turnover Denominator": st.column_config.NumberColumn("Turnover Denominator", format="%.0f"),
        "Current Float Turnover": st.column_config.NumberColumn("Current Float Turnover", format="%.1f%%"),
        "Average Daily Float Turnover": st.column_config.NumberColumn("Avg Daily Float Turnover", format="%.1f%%"),
        "RS vs Theme": st.column_config.NumberColumn("RS vs Theme", format="%.1f%%"),
        "RS vs Peer Median": st.column_config.NumberColumn("RS vs Peer Median", format="%.1f%%"),
        "RS vs Sector": st.column_config.NumberColumn("RS vs Sector", format="%.1f%%"),
        "RS vs Industry": st.column_config.NumberColumn("RS vs Industry", format="%.1f%%"),
    }


def _select_overlay_history_tickers(results_df: pd.DataFrame, target_ticker: str | None, limit: int) -> list[str]:
    selected: list[str] = []

    def add(symbol: str | None) -> None:
        ticker = normalize_ticker(symbol or "")
        if ticker and ticker not in selected:
            selected.append(ticker)

    add(target_ticker)
    for _, row in results_df.iterrows():
        bucket = str(row.get("Bucket", ""))
        if "Benchmark" in bucket:
            add(str(row.get("Ticker", "")))
    ranked = results_df.sort_values("Momentum Trend Score", ascending=False)
    for _, row in ranked.iterrows():
        add(str(row.get("Ticker", "")))
        if len(selected) >= limit:
            break
    return selected[:limit]


def _approx_combined_overlay_score(row) -> float:
    momentum = _number(row.get("Momentum Trend Score")) or 0.0
    rs = _number(row.get("RS Percentile"))
    rs_score = (rs * 100) if rs is not None else 50.0
    volume = (_number(row.get("Volume Confirmation Score")) or 0.0) * 10
    risk = (_number(row.get("Risk Volatility Score")) or 0.0) * 10
    score = momentum * 0.45 + rs_score * 0.25 + volume * 0.15 + risk * 0.15
    return float(max(0, min(100, score)))


def _approx_overlay_history(frame: pd.DataFrame, timeframe: str) -> pd.Series:
    clean = _clean_ohlcv(frame)
    if clean.shape[0] < 80:
        return pd.Series(dtype=float)
    close = clean["Close"].astype(float)
    volume = clean["Volume"].astype(float) if "Volume" in clean.columns else pd.Series(index=clean.index, dtype=float)
    window = SCREENER_TIMEFRAME_WINDOWS.get(timeframe, 21)

    ret = close.pct_change(window)
    momentum_score = (50 + ret.clip(-0.30, 0.30) * 120).clip(0, 100)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    trend_score = (
        (close > sma20).astype(float) * 20
        + (close > sma50).astype(float) * 25
        + (close > sma200).astype(float) * 25
        + (sma20 > sma50).astype(float) * 15
        + (sma50 > sma200).astype(float) * 15
    )
    rvol = volume / volume.rolling(20).mean()
    volume_score = (50 + (rvol - 1).clip(-1, 1.5) * 25).clip(0, 100)
    realized_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    risk_score = (100 - realized_vol.fillna(0.35).clip(0, 1.2) * 70).clip(0, 100)
    overlay = trend_score * 0.40 + momentum_score * 0.30 + volume_score * 0.15 + risk_score * 0.15
    return overlay.dropna().clip(0, 100).tail(252)


def _render_screener_control_impact(controls: dict, scored_count: int, filtered_count: int, shown_count: int) -> None:
    st.markdown("**How the current controls affect these results**")
    impact_rows = [
        {
            "Control": "Screener mode",
            "Current": controls.get("mode", "N/A"),
            "Affects": "Which tickers enter the universe before scoring.",
            "Where reflected": "Universe count, ranking table, bucket tables, and charts.",
        },
        {
            "Control": "Theme group",
            "Current": controls.get("theme", "N/A"),
            "Affects": "Theme peers and theme median used for RS vs Theme.",
            "Where reflected": "RS vs Theme, Theme Peers bucket, target summary.",
        },
        {
            "Control": "Benchmark",
            "Current": controls.get("benchmark", "N/A"),
            "Affects": "Primary relative-strength benchmark.",
            "Where reflected": "RS vs Benchmark, benchmark filter, score, bucket charts.",
        },
        {
            "Control": "Market benchmark",
            "Current": controls.get("market_benchmark", "N/A"),
            "Affects": "Broad-market comparison.",
            "Where reflected": f"RS vs {controls.get('market_benchmark', 'Market')} column and relative-strength score context.",
        },
        {
            "Control": "Minimum dollar volume",
            "Current": _format_dollar(controls.get("min_dollar_volume")),
            "Affects": "Liquidity filter: latest price times latest volume.",
            "Where reflected": f"Filters {scored_count - filtered_count} names before Top N.",
        },
        {
            "Control": "Top N",
            "Current": str(controls.get("top_n", "N/A")),
            "Affects": "How many filtered names are displayed.",
            "Where reflected": f"Showing {shown_count} of {filtered_count} filtered names.",
        },
    ]
    with st.expander("Controls impact", expanded=False):
        st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)


def _render_selected_ticker_preview(df: pd.DataFrame, timeframe: str, benchmark: str) -> None:
    st.subheader("Selected Ticker Preview")
    st.caption("Use this drilldown to inspect one row from the current results table. It does not re-run the screener by itself.")
    selected = st.selectbox("Select ticker", df["Ticker"].tolist(), key="screener_selected_preview_ticker")
    row = df.loc[df["Ticker"] == selected].iloc[0]
    st.markdown(f"**{selected} - {row['Label']}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rank", f"{int(row['Rank'])} / {len(df)}")
    c2.metric("Score", f"{row['Momentum Trend Score']:.0f}/100")
    c3.metric(f"{timeframe} Return", _format_pct(row["Timeframe Return"]))
    c4.metric(f"RS vs {benchmark}", _format_pct(row["RS vs Benchmark"]))
    c5.metric("RS vs Theme", _format_pct(row["RS vs Theme"]))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Bucket", str(row.get("Bucket", "Universe")))
    c7.metric("Price", _format_price(row.get("Price")))
    c8.metric("Dollar Volume", _format_dollar(row.get("Dollar Volume")))
    c9.metric("ATR Risk", _format_pct(row.get("ATR %")))
    c10, c11, c12 = st.columns(3)
    turnover_label = str(row.get("Turnover Type", "Turnover"))
    c10.metric("Avg Daily Turnover", _format_pct(row.get("Average Daily Turnover")))
    c11.metric(f"Current {turnover_label}", _format_pct(row.get("Turnover %")))
    c12.metric("Turnover Source", str(row.get("Turnover Source", "Unavailable")))

    score_columns = [
        "Price Trend Score",
        "Momentum Score",
        "Relative Strength Score",
        "Volume Confirmation Score",
        "Risk Volatility Score",
    ]
    score_df = pd.DataFrame(
        {
            "Component": [label.replace(" Score", "") for label in score_columns],
            "Points": [_number(row.get(column)) or 0.0 for column in score_columns],
        }
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=score_df["Component"],
                y=score_df["Points"],
                marker_color=["#22d3ee", "#60a5fa", "#a78bfa", "#22c55e", "#f59e0b"],
                text=[f"{value:.0f}" for value in score_df["Points"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"{selected} score breakdown",
        height=330,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=55, b=40),
        yaxis=dict(title="Points", range=[0, max(30, float(score_df["Points"].max()) + 8)]),
        xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch", key=f"screener_preview_score_breakdown_{selected}")

    relative_rows = [
        ("RS vs Target", row.get("RS vs Target")),
        (f"RS vs {benchmark}", row.get("RS vs Benchmark")),
        ("RS vs Market", row.get("RS vs SPY")),
        ("RS vs Theme", row.get("RS vs Theme")),
        ("RS vs Peer Median", row.get("RS vs Peer Median")),
        ("RS vs Sector", row.get("RS vs Sector")),
        ("RS vs Industry", row.get("RS vs Industry")),
    ]
    relative_df = pd.DataFrame(
        [
            {"Comparison": label, "Gap": _format_pct(value), "Read": _relative_read(value)}
            for label, value in relative_rows
            if _number(value) is not None
        ]
    )
    if not relative_df.empty:
        st.dataframe(relative_df, use_container_width=True, hide_index=True)

    st.info(_quick_conclusion(row=row, timeframe=timeframe, benchmark=benchmark))
    if st.button("Open in MR-1 Dashboard", type="primary", key=f"open_screener_{selected}"):
        st.session_state["ticker_search"] = selected
        st.session_state["dashboard_tab_selector"] = "Overview"
        st.rerun()


def _display_columns(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    columns = ["Rank", "Ticker"]
    if include_target:
        columns.append("Bucket")
    columns.extend(
        [
            "Momentum Trend Score",
            "Label",
            "Timeframe Return",
        ]
    )
    if include_target:
        columns.append("RS vs Target")
    columns.extend(
        [
            "RS vs Benchmark",
            "RS vs SPY",
            "RS vs Theme",
            "Trend",
            "Volume",
            "Risk",
        ]
    )
    display = df[columns].copy()
    for column in ["Timeframe Return", "RS vs Target", "RS vs Benchmark", "RS vs SPY", "RS vs Theme"]:
        if column not in display.columns:
            continue
        display[column] = display[column] * 100
    return display


def _advanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Bucket",
        "Bucket Source",
        "Override Applied",
        "Override Mode",
        "Company",
        "Sector",
        "Industry",
        "Price",
        "Dollar Volume",
        "Shares Float",
        "Turnover %",
        "Average Daily Turnover",
        "5D Turnover",
        "Turnover Type",
        "Turnover Source",
        "Turnover Denominator",
        "Average Daily Float Turnover",
        "Current Float Turnover",
        "RVOL 20D",
        "ATR %",
        "Distance 50D SMA",
        "Distance 200D SMA",
        "RS vs Target",
        "RS vs SPY",
        "RS vs Sector",
        "RS vs Industry",
        "Volume Percentile",
        "Recent Drawdown",
        "Warnings",
    ]
    existing = [column for column in columns if column in df.columns]
    display = df[existing].copy()
    for column in [
        "ATR %",
        "Distance 50D SMA",
        "Distance 200D SMA",
        "RS vs SPY",
        "RS vs Target",
        "RS vs Sector",
        "RS vs Industry",
        "Volume Percentile",
        "Recent Drawdown",
        "Turnover %",
        "Average Daily Turnover",
        "5D Turnover",
    ]:
        if column in display.columns:
            display[column] = display[column] * 100
    return display


def _column_config(benchmark: str = "Benchmark", market_benchmark: str = "Market") -> dict:
    return {
        "Momentum Trend Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Timeframe Return": st.column_config.NumberColumn("Timeframe Return", format="%.1f%%"),
        "RS vs Target": st.column_config.NumberColumn("RS vs Target", format="%.1f%%"),
        "RS vs Benchmark": st.column_config.NumberColumn(f"RS vs {benchmark}", format="%.1f%%"),
        "RS vs SPY": st.column_config.NumberColumn(f"RS vs {market_benchmark}", format="%.1f%%"),
        "RS vs Theme": st.column_config.NumberColumn("RS vs Theme", format="%.1f%%"),
    }


def _advanced_column_config() -> dict:
    return {
        "Override Applied": st.column_config.CheckboxColumn("Override Applied"),
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Dollar Volume": st.column_config.NumberColumn("Dollar Volume", format="$%.0f"),
        "Shares Float": st.column_config.NumberColumn("Shares Float", format="%.0f"),
        "Turnover %": st.column_config.NumberColumn("Turnover %", format="%.1f%%"),
        "Average Daily Turnover": st.column_config.NumberColumn("Avg Daily Turnover", format="%.1f%%"),
        "5D Turnover": st.column_config.NumberColumn("5D Turnover", format="%.1f%%"),
        "Turnover Denominator": st.column_config.NumberColumn("Turnover Denominator", format="%.0f"),
        "Average Daily Float Turnover": st.column_config.NumberColumn("Avg Daily Float Turnover", format="%.1f%%"),
        "Current Float Turnover": st.column_config.NumberColumn("Current Float Turnover", format="%.1f%%"),
        "RVOL 20D": st.column_config.NumberColumn("RVOL", format="%.2f"),
        "ATR %": st.column_config.NumberColumn("ATR %", format="%.1f%%"),
        "Distance 50D SMA": st.column_config.NumberColumn("Dist 50D", format="%.1f%%"),
        "Distance 200D SMA": st.column_config.NumberColumn("Dist 200D", format="%.1f%%"),
        "RS vs Target": st.column_config.NumberColumn("RS vs Target", format="%.1f%%"),
        "RS vs SPY": st.column_config.NumberColumn("RS vs Market", format="%.1f%%"),
        "RS vs Sector": st.column_config.NumberColumn("RS vs Sector", format="%.1f%%"),
        "RS vs Industry": st.column_config.NumberColumn("RS vs Industry", format="%.1f%%"),
        "Volume Percentile": st.column_config.NumberColumn("Volume Percentile", format="%.0f%%"),
        "Recent Drawdown": st.column_config.NumberColumn("Recent Drawdown", format="%.1f%%"),
    }


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in frame.columns]
    return frame[columns].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])


def _timeframe_return(frame: pd.DataFrame | None, window: int) -> float:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return 0.0
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    return _period_return(close, window)


def _period_return(close: pd.Series, window: int) -> float:
    if close.shape[0] <= window:
        return 0.0
    previous = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if previous == 0:
        return 0.0
    return latest / previous - 1


def _last_sma(series: pd.Series, window: int) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.shape[0] < window:
        return np.nan
    return float(numeric.rolling(window).mean().iloc[-1])


def _distance(price: float, average: float) -> float:
    if not average or not math.isfinite(average):
        return np.nan
    return price / average - 1


def _volume_percentile(volume: pd.Series) -> float:
    numeric = pd.to_numeric(volume, errors="coerce").dropna().tail(252)
    if numeric.shape[0] < 20:
        return np.nan
    return float(numeric.rank(pct=True).iloc[-1])


def _atr_pct(frame: pd.DataFrame, window: int = 14) -> float:
    if not {"High", "Low", "Close"}.issubset(frame.columns):
        return np.nan
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean().iloc[-1]
    latest = close.iloc[-1]
    if not latest:
        return np.nan
    return float(atr / latest)


def _score_label(score: float) -> str:
    if score >= 80:
        return "Momentum Leader"
    if score >= 65:
        return "Strong Candidate"
    if score >= 50:
        return "Watchlist Candidate"
    if score >= 35:
        return "Mixed / Weak"
    return "Avoid / Downtrend"


def _trend_status(row) -> str:
    score = _number(row.get("Price Trend Score")) or 0.0
    if score >= 25:
        return "Clean Uptrend"
    if score >= 18:
        return "Uptrend"
    if score >= 10:
        return "Mixed"
    return "Downtrend"


def _volume_status(row) -> str:
    if row.get("Distribution Warning"):
        return "Distribution"
    score = _number(row.get("Volume Confirmation Score")) or 0.0
    if score >= 10:
        return "Confirmed"
    if score >= 7:
        return "Normal"
    if score >= 3:
        return "Weak"
    return "Not Confirmed"


def _risk_status(row) -> str:
    score = _number(row.get("Risk Volatility Score")) or 0.0
    if score >= 10:
        return "Clean"
    if score >= 6:
        return "Tradable"
    if score >= 3:
        return "Extended"
    return "High Risk"


def _row_warnings(row) -> str:
    warnings = []
    if row.get("Distribution Warning"):
        warnings.append("distribution")
    if (_number(row.get("Risk Volatility Score")) or 0.0) < 6:
        warnings.append("volatile/extended")
    if (_number(row.get("Volume Confirmation Score")) or 0.0) < 7:
        warnings.append("weak volume")
    return ", ".join(warnings)


def _quick_conclusion(row, timeframe: str, benchmark: str) -> str:
    ticker = row["Ticker"]
    outperform_benchmark = (_number(row.get("RS vs Benchmark")) or 0.0) > 0
    outperform_theme = (_number(row.get("RS vs Theme")) or 0.0) > 0
    parts = [
        f"{ticker} is a {row['Label']} over the selected {timeframe} timeframe.",
        f"Trend is {str(row['Trend']).lower()}, volume is {str(row['Volume']).lower()}, and risk is {str(row['Risk']).lower()}.",
    ]
    if outperform_benchmark and outperform_theme:
        parts.insert(1, f"It is outperforming both {benchmark} and the selected universe/theme.")
    elif outperform_benchmark:
        parts.insert(1, f"It is outperforming {benchmark}, but theme-relative strength is not fully confirmed.")
    elif outperform_theme:
        parts.insert(1, "It is beating the selected universe/theme, but not the benchmark.")
    else:
        parts.insert(1, f"It is not outperforming {benchmark} or the selected universe/theme.")
    return " ".join(parts)


def _format_pct(value) -> str:
    number = _number(value)
    if number is None or not math.isfinite(number):
        return "N/A"
    return f"{number * 100:.1f}%"


def _format_price(value) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    return f"${number:,.2f}"


def _format_dollar(value) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"${number:,.0f}"


def _relative_read(value) -> str:
    number = _number(value)
    if number is None:
        return "Unavailable"
    if number >= 0.03:
        return "Clear leadership"
    if number > 0:
        return "Slight leadership"
    if number <= -0.03:
        return "Clear lag"
    return "Slight lag"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
