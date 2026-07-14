from __future__ import annotations

import math
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import (
    SCREENER_CONFIG,
    SCREENER_PEER_MAP,
    SCREENER_TARGET_TICKER_CONFIG,
    SCREENER_THEME_GROUPS,
    SCREENER_WATCHLISTS,
    SECTOR_ETF_MAP,
)
from data import normalize_ticker
from finviz_fetcher import remove_dead_local_proxy


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


def calculate_screener_features(
    ticker_data: dict[str, pd.DataFrame],
    benchmark_data: dict[str, pd.DataFrame],
    theme_tickers: list[str],
    timeframe: str,
    metadata: dict | None = None,
) -> pd.DataFrame:
    del metadata
    window = SCREENER_TIMEFRAME_WINDOWS.get(timeframe, 21)
    benchmark_symbol = str(benchmark_data.get("benchmark_symbol", SCREENER_CONFIG["default_benchmark"]))
    market_symbol = str(benchmark_data.get("market_symbol", SCREENER_CONFIG["default_market_benchmark"]))
    target_symbol = normalize_ticker(str(benchmark_data.get("target_symbol", "")))
    include_benchmarks = bool(benchmark_data.get("include_benchmarks", False))
    buckets = benchmark_data.get("buckets", {}) or {}
    benchmark_return = _timeframe_return(benchmark_data.get("benchmark"), window)
    market_return = _timeframe_return(benchmark_data.get("market"), window)
    target_return = _timeframe_return(ticker_data.get(target_symbol), window) if target_symbol else np.nan

    rows: list[dict] = []
    for ticker, frame in ticker_data.items():
        if not include_benchmarks and ticker in {benchmark_symbol, market_symbol}:
            continue
        row = _build_feature_row(
            ticker=ticker,
            frame=frame,
            timeframe=timeframe,
            window=window,
            benchmark_return=benchmark_return,
            market_return=market_return,
        )
        if row:
            row["Bucket"] = _bucket_label(buckets.get(ticker, []))
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
    features["Momentum Percentile"] = features["Timeframe Return"].rank(pct=True)
    features["RS Percentile"] = features["RS Composite"].rank(pct=True)
    features["Theme Percentile"] = features["Timeframe Return"].rank(pct=True)
    features["Sector"] = None
    features["Industry"] = None
    features["Company"] = None
    features["RS vs Sector"] = np.nan
    features["RS vs Industry"] = np.nan
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

    if not controls["run"]:
        st.info("Set the controls, then click **Run Screener**.")
        return

    period = SCREENER_PERIODS.get(controls["timeframe"], "2y")
    all_symbols = normalize_ticker_list([*universe_tickers, controls["benchmark"], controls["market_benchmark"]])
    with st.spinner(f"Screening {len(universe_tickers)} tickers..."):
        all_data = fetch_screener_data(all_symbols, period=period)

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
        },
        theme_tickers=controls.get("theme_tickers", universe_tickers),
        timeframe=controls["timeframe"],
    )
    scored = calculate_momentum_trend_score(features)
    if controls.get("mode") == "Ticker Comparison":
        _render_target_relative_summary(
            target_ticker=controls["target_ticker"],
            scored=scored,
            benchmark=controls["benchmark"],
            market_benchmark=controls["market_benchmark"],
            theme_name=controls.get("detected_theme"),
        )
    filtered = _apply_screener_filters(scored, controls)
    sorted_df = _sort_screener(filtered, controls["sort_by"]).head(int(controls["top_n"]))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Screened", f"{len(scored)}")
    col_b.metric("Shown", f"{len(sorted_df)}")
    col_c.metric("Skipped", f"{len(skipped)}")
    if skipped:
        st.caption(f"Skipped missing data: {', '.join(skipped[:20])}{'...' if len(skipped) > 20 else ''}")

    if sorted_df.empty:
        st.warning("No tickers passed the selected filters.")
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
            column_config=_column_config(),
        )
    else:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config(),
        )

    with st.expander("Advanced screener columns", expanded=False):
        st.dataframe(
            _advanced_columns(sorted_df),
            use_container_width=True,
            hide_index=True,
            column_config=_advanced_column_config(),
        )

    _render_selected_ticker_preview(sorted_df, timeframe=controls["timeframe"], benchmark=controls["benchmark"])


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
            )
            watchlist = st.selectbox("Watchlist", list(SCREENER_WATCHLISTS.keys()), index=0)
        with col2:
            benchmark = normalize_ticker(
                st.text_input("Benchmark", value=str(preset.get("benchmark", target_universe.get("benchmark") or SCREENER_CONFIG["default_benchmark"])))
            )
            market_benchmark = normalize_ticker(
                st.text_input("Market benchmark", value=str(preset.get("market_benchmark", target_universe.get("market_benchmark") or SCREENER_CONFIG["default_market_benchmark"])))
            )
            timeframe_options = list(SCREENER_TIMEFRAME_WINDOWS.keys())
            timeframe = st.selectbox(
                "Timeframe",
                timeframe_options,
                index=timeframe_options.index(str(preset.get("timeframe", SCREENER_CONFIG["default_timeframe"]))),
            )
        with col3:
            filters = preset.get("filters", {}) if isinstance(preset.get("filters"), dict) else {}
            min_price = st.number_input("Minimum price", min_value=0.0, value=float(filters.get("min_price", SCREENER_CONFIG["default_min_price"])), step=1.0)
            min_dollar_volume = st.number_input(
                "Minimum dollar volume",
                min_value=0.0,
                value=float(filters.get("min_dollar_volume", SCREENER_CONFIG["default_min_dollar_volume"])),
                step=1_000_000.0,
            )
            top_n = st.number_input("Top N", min_value=1, max_value=int(SCREENER_CONFIG["max_tickers"]), value=int(filters.get("top_n", SCREENER_CONFIG["default_top_n"])))

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
    if mode == "Ticker Comparison":
        buckets.setdefault(target_ticker, ["Target"])
        buckets.setdefault(benchmark or str(SCREENER_CONFIG["default_benchmark"]), ["Benchmark"])
        buckets.setdefault(market_benchmark or str(SCREENER_CONFIG["default_market_benchmark"]), ["Market Benchmark"])
        for ticker in comparison_tickers:
            buckets.setdefault(ticker, ["Manual Override"])
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
    )
    with st.expander("Save / load screener preset", expanded=False):
        uploaded = st.file_uploader("Load preset JSON", type=["json"], key="screener_preset_upload")
        if uploaded is not None:
            try:
                st.session_state["screener_loaded_preset"] = json.loads(uploaded.getvalue().decode("utf-8"))
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
        "comparison_tickers": comparison_tickers,
        "include_benchmarks": mode == "Ticker Comparison" and bool(SCREENER_TARGET_TICKER_CONFIG.get("include_benchmarks_in_results", True)),
        "theme_tickers": normalize_ticker_list(SCREENER_THEME_GROUPS.get(target_universe.get("theme") or theme, [])),
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
    st.info(
        f"**{target} comparison universe** | "
        f"Detected theme: {universe.get('theme') or 'Not detected'} | "
        f"Sector: {universe.get('sector') or 'Unavailable'} | "
        f"Industry: {universe.get('industry') or 'Unavailable'} | "
        f"Benchmark: {controls['benchmark']} | Market: {controls['market_benchmark']}"
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
    }


def _bucket_label(buckets: list[str] | tuple[str, ...] | None) -> str:
    values = [str(bucket) for bucket in (buckets or []) if str(bucket)]
    return " / ".join(values) if values else "Universe"


def _build_feature_row(
    ticker: str,
    frame: pd.DataFrame,
    timeframe: str,
    window: int,
    benchmark_return: float,
    market_return: float,
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


def _render_selected_ticker_preview(df: pd.DataFrame, timeframe: str, benchmark: str) -> None:
    st.subheader("Selected Ticker Preview")
    selected = st.selectbox("Select ticker", df["Ticker"].tolist())
    row = df.loc[df["Ticker"] == selected].iloc[0]
    st.markdown(f"**{selected} - {row['Label']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{row['Momentum Trend Score']:.0f}/100")
    c2.metric(f"{timeframe} Return", _format_pct(row["Timeframe Return"]))
    c3.metric(f"RS vs {benchmark}", _format_pct(row["RS vs Benchmark"]))
    c4.metric("RS vs Theme", _format_pct(row["RS vs Theme"]))
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
        "Company",
        "Sector",
        "Industry",
        "Price",
        "Dollar Volume",
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
    display = df[columns].copy()
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
    ]:
        display[column] = display[column] * 100
    return display


def _column_config() -> dict:
    return {
        "Momentum Trend Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Timeframe Return": st.column_config.NumberColumn("Timeframe Return", format="%.1f%%"),
        "RS vs Target": st.column_config.NumberColumn("RS vs Target", format="%.1f%%"),
        "RS vs Benchmark": st.column_config.NumberColumn("RS vs Benchmark", format="%.1f%%"),
        "RS vs SPY": st.column_config.NumberColumn("RS vs SPY", format="%.1f%%"),
        "RS vs Theme": st.column_config.NumberColumn("RS vs Theme", format="%.1f%%"),
    }


def _advanced_column_config() -> dict:
    return {
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Dollar Volume": st.column_config.NumberColumn("Dollar Volume", format="$%.0f"),
        "RVOL 20D": st.column_config.NumberColumn("RVOL", format="%.2f"),
        "ATR %": st.column_config.NumberColumn("ATR %", format="%.1f%%"),
        "Distance 50D SMA": st.column_config.NumberColumn("Dist 50D", format="%.1f%%"),
        "Distance 200D SMA": st.column_config.NumberColumn("Dist 200D", format="%.1f%%"),
        "RS vs Target": st.column_config.NumberColumn("RS vs Target", format="%.1f%%"),
        "RS vs SPY": st.column_config.NumberColumn("RS vs SPY", format="%.1f%%"),
        "RS vs Sector": st.column_config.NumberColumn("RS vs Sector", format="%.1f%%"),
        "RS vs Industry": st.column_config.NumberColumn("RS vs Industry", format="%.1f%%"),
        "Volume Percentile": st.column_config.NumberColumn("Volume Percentile", format="%.0f%%"),
        "Recent Drawdown": st.column_config.NumberColumn("Recent Drawdown", format="%.1f%%"),
    }


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
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


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
