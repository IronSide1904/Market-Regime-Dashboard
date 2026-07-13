from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import SCREENER_CONFIG, SCREENER_THEME_GROUPS, SCREENER_WATCHLISTS, SECTOR_ETF_MAP
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
    benchmark_return = _timeframe_return(benchmark_data.get("benchmark"), window)
    market_return = _timeframe_return(benchmark_data.get("market"), window)

    rows: list[dict] = []
    for ticker, frame in ticker_data.items():
        if ticker in {benchmark_symbol, market_symbol}:
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
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    features = pd.DataFrame(rows)
    theme_returns = features.loc[features["Ticker"].isin(theme_tickers), "Timeframe Return"]
    theme_median = float(theme_returns.median()) if not theme_returns.dropna().empty else float(features["Timeframe Return"].median())
    features["Theme Median Return"] = theme_median
    features["RS vs Theme"] = features["Timeframe Return"] - theme_median
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
        "Lightweight discovery tool for clean momentum versus a benchmark, SPY, and the selected theme/universe. "
        "It does not run full MR-1 analysis for every ticker."
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
        },
        theme_tickers=universe_tickers,
        timeframe=controls["timeframe"],
    )
    scored = calculate_momentum_trend_score(features)
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

    st.dataframe(
        _display_columns(sorted_df),
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
    source_options = ["Theme Group", "Watchlist", "Manual Ticker List", "Sector ETF List"]
    with st.form("screener_controls"):
        col1, col2, col3 = st.columns(3)
        with col1:
            universe_source = st.selectbox("Universe source", source_options, index=0)
            theme = st.selectbox("Theme group", list(SCREENER_THEME_GROUPS.keys()), index=0)
            watchlist = st.selectbox("Watchlist", list(SCREENER_WATCHLISTS.keys()), index=0)
        with col2:
            benchmark = normalize_ticker(
                st.text_input("Benchmark", value=str(SCREENER_CONFIG["default_benchmark"]))
            )
            market_benchmark = normalize_ticker(
                st.text_input("Market benchmark", value=str(SCREENER_CONFIG["default_market_benchmark"]))
            )
            timeframe_options = list(SCREENER_TIMEFRAME_WINDOWS.keys())
            timeframe = st.selectbox(
                "Timeframe",
                timeframe_options,
                index=timeframe_options.index(str(SCREENER_CONFIG["default_timeframe"])),
            )
        with col3:
            min_price = st.number_input("Minimum price", min_value=0.0, value=float(SCREENER_CONFIG["default_min_price"]), step=1.0)
            min_dollar_volume = st.number_input(
                "Minimum dollar volume",
                min_value=0.0,
                value=float(SCREENER_CONFIG["default_min_dollar_volume"]),
                step=1_000_000.0,
            )
            top_n = st.number_input("Top N", min_value=1, max_value=int(SCREENER_CONFIG["max_tickers"]), value=int(SCREENER_CONFIG["default_top_n"]))

        manual_tickers = st.text_area(
            "Manual ticker input",
            value="NVDA, AMD, AVGO, MRVL, MU",
            help="Used when Universe source is Manual Ticker List. Commas, spaces, and new lines are supported.",
        )

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0, step=5)
            only_above_50d = st.checkbox("Only above 50D SMA", value=False)
        with f2:
            only_above_200d = st.checkbox("Only above 200D SMA", value=False)
            only_outperforming_benchmark = st.checkbox("Only outperforming benchmark", value=True)
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
                    "RS vs Theme",
                    "Volume Confirmation Score",
                    "ATR %",
                    "Distance 50D SMA",
                    "Distance 200D SMA",
                ],
            )

        run = st.form_submit_button("Run Screener", type="primary")

    if universe_source == "Sector ETF List":
        st.caption("Sector ETF List uses sector ETFs as a lightweight sector universe. Stock sector/industry screening requires Finviz metadata and is not run by default.")

    return {
        "universe_source": universe_source,
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
    source = controls["universe_source"]
    if source == "Manual Ticker List":
        return normalize_ticker_list(controls["manual_tickers"])
    if source == "Watchlist":
        return normalize_ticker_list(SCREENER_WATCHLISTS.get(controls["watchlist"], []))
    if source == "Sector ETF List":
        return normalize_ticker_list(SECTOR_ETF_MAP.values())
    return normalize_ticker_list(SCREENER_THEME_GROUPS.get(controls["theme"], []))


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


def _display_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Rank",
        "Ticker",
        "Momentum Trend Score",
        "Label",
        "Timeframe Return",
        "RS vs Benchmark",
        "RS vs Theme",
        "Trend",
        "Volume",
        "Risk",
    ]
    display = df[columns].copy()
    for column in ["Timeframe Return", "RS vs Benchmark", "RS vs Theme"]:
        display[column] = display[column] * 100
    return display


def _advanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "Price",
        "Dollar Volume",
        "RVOL 20D",
        "ATR %",
        "Distance 50D SMA",
        "Distance 200D SMA",
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
        "RS vs Benchmark": st.column_config.NumberColumn("RS vs Benchmark", format="%.1f%%"),
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
