from __future__ import annotations

from dataclasses import replace
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_TITLE,
    DEBUG_MODE,
    DEFAULT_BENCHMARK,
    DEFAULT_TIMEFRAME_PRESET,
    DEFAULT_TICKER,
    REGIME_RULES,
    RELATIVE_CONTEXT_CONFIG,
    RELATIVE_TREND_QUALITY_CONFIG,
    SENSITIVITY_LOOKBACKS,
    SWING_TIMEFRAMES,
    SWING_TIMEFRAME_TO_VOLATILITY_PRESET,
    TIMEFRAME_PRESETS,
    TIMEFRAME_TO_VOLUME_PRESET,
    TIMEFRAMES,
    VOLUME_CONTEXT_CONFIG,
    get_swing_timeframe_profile,
    get_timeframe_score_profile,
)
from data import fetch_market_data, get_benchmark_ohlcv, get_comparison_benchmark, get_ticker_metadata, normalize_ticker
from events import EventContextResult, build_event_context
from hmm_model import HMMResult, build_hmm_result
from indicators import calculate_indicators
from metadata import (
    ASSET_CONTEXTS,
    INDUSTRY_PROXIES,
    SECTOR_ETFS,
    SUB_INDUSTRY_TICKERS,
    THEME_TICKERS,
    get_asset_context,
)
from options_data import OptionsVolatilityResult, build_options_volatility_context
from regime_persistence import (
    RegimePersistenceResult,
    analyze_regime_persistence,
    build_peer_persistence_table,
)
from relative_context import analyze_relative_context
from relative_trend_quality import calculate_clean_relative_trend_score
from scoring import (
    apply_relative_context_adjustment,
    backtest_metrics,
    latest_signal_breakdown,
    main_drivers,
    run_backtest,
    score_history,
    get_exposure,
    get_regime,
)
from swing import SwingResult, build_swing_result
from volume import get_timeframe_config


KNOWN_ETF_SYMBOLS = {
    *SECTOR_ETFS.values(),
    *INDUSTRY_PROXIES.values(),
    *THEME_TICKERS.values(),
    *SUB_INDUSTRY_TICKERS.values(),
    *(context.sector_etf for context in ASSET_CONTEXTS.values()),
    *(context.industry_proxy for context in ASSET_CONTEXTS.values()),
    *(context.theme_ticker for context in ASSET_CONTEXTS.values()),
    *(context.sub_industry_ticker for context in ASSET_CONTEXTS.values()),
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "RSP",
    "MDY",
    "VTI",
    "ARKK",
    "XLK",
    "XLU",
    "SMH",
    "SOXX",
}

AUTO_VOLUME_PRESET_OPTION = "Auto (match Timeframe)"
AUTO_VOLATILITY_PRESET_OPTION = "Auto (match Swing timeframe)"


def render_dashboard() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="MR1", layout="wide")
    _render_styles()

    ticker, last_updated_slot = _render_top_bar()
    (
        benchmark,
        timeframe_label,
        swing_timeframe,
        sensitivity,
        volume_preset_choice,
        volatility_preset_choice,
        peer_override_mode,
        peer_override_tickers,
    ) = _render_sidebar()
    score_profile = get_timeframe_score_profile(timeframe_label, sensitivity)
    lookbacks = score_profile["lookbacks"]
    swing_profile = get_swing_timeframe_profile(swing_timeframe)
    volume_preset, volatility_preset = _resolve_volume_volatility_presets(
        timeframe_label=timeframe_label,
        swing_timeframe=swing_timeframe,
        volume_preset_choice=volume_preset_choice,
        volatility_preset_choice=volatility_preset_choice,
    )
    volume_timeframe_config = get_timeframe_config(volume_preset)
    volatility_timeframe_config = get_timeframe_config(volatility_preset)

    with st.spinner(f"Loading {ticker} regime dashboard..."):
        market_data = fetch_market_data(
            ticker=ticker,
            benchmark=benchmark,
            period=TIMEFRAMES[timeframe_label],
        )

    hard_errors = [
        warning
        for warning in market_data.warnings
        if "Ticker not found" in warning or "is missing from the downloaded data" in warning
    ]
    if hard_errors:
        _render_last_updated(last_updated_slot, None)
        for warning in hard_errors:
            st.error(warning)
        st.stop()

    clean_close = market_data.close.dropna()
    indicators = calculate_indicators(clean_close, lookbacks=lookbacks)
    if indicators.empty:
        _render_last_updated(last_updated_slot, None)
        st.error("Ticker not found or data unavailable.")
        st.stop()

    ticker_metadata = get_ticker_metadata(ticker)
    shares_float = ticker_metadata.get("shares_float")
    scored = score_history(
        indicators,
        ticker_ohlcv=market_data.ticker_ohlcv,
        shares_float=shares_float,
        volume_timeframe_config=volume_timeframe_config,
    )
    display_scored = _filter_scored_by_timeframe(scored, timeframe_label)
    signals = latest_signal_breakdown(scored, ticker=ticker, benchmark=benchmark)
    positive_driver, negative_driver = main_drivers(signals)
    latest = scored.iloc[-1]
    context = _apply_peer_override(
        context=get_asset_context(ticker=ticker, benchmark=benchmark),
        ticker=ticker,
        mode=peer_override_mode,
        override_tickers=peer_override_tickers,
    )

    comparison_benchmark = get_comparison_benchmark(
        ticker=ticker,
        metadata=ticker_metadata,
        user_benchmark=benchmark,
    )
    if RELATIVE_CONTEXT_CONFIG.get("enabled", True):
        with st.spinner(f"Loading {ticker} relative context vs {comparison_benchmark}..."):
            benchmark_ohlcv = _load_benchmark_ohlcv(comparison_benchmark, period=TIMEFRAMES[timeframe_label])
            relative_context = analyze_relative_context(
                ticker=ticker,
                benchmark=comparison_benchmark,
                ticker_ohlcv=market_data.ticker_ohlcv,
                benchmark_ohlcv=benchmark_ohlcv,
                ticker_volume_metrics=_ticker_relative_volume_metrics(latest),
                benchmark_volume_metrics=_benchmark_relative_volume_metrics(benchmark_ohlcv),
            )
    else:
        benchmark_ohlcv = pd.DataFrame()
        relative_context = {
            "available": False,
            "ticker": ticker.upper(),
            "benchmark": comparison_benchmark.upper(),
            "relationship_status": "Unavailable",
            "score_adjustment": 0,
            "confidence": "Low",
            "interpretation": "Relative Context is disabled.",
            "warnings": ["Relative Context is disabled."],
            "history": pd.DataFrame(),
        }
    relative_adjusted_score = int(round(apply_relative_context_adjustment(float(latest["MR-1 Score"]), relative_context)))
    relative_adjusted_regime = get_regime(relative_adjusted_score)
    relative_adjusted_exposure = get_exposure(relative_adjusted_regime)
    ticker_quality_ohlcv = market_data.ticker_ohlcv.copy()
    benchmark_quality_ohlcv = benchmark_ohlcv.copy()
    ticker_quality_ohlcv.attrs["symbol"] = ticker.upper()
    benchmark_quality_ohlcv.attrs["symbol"] = comparison_benchmark.upper()
    clean_relative_trend = calculate_clean_relative_trend_score(
        ticker_df=ticker_quality_ohlcv,
        benchmark_df=benchmark_quality_ohlcv,
        timeframe=timeframe_label,
        volume_context=_clean_relative_trend_volume_context(latest),
        asset_type=_relative_trend_asset_type(ticker=ticker, metadata=ticker_metadata),
    )

    _render_last_updated(last_updated_slot, scored.index[-1])
    for warning in market_data.warnings:
        st.warning(warning)

    with st.spinner("Loading expanded HMM market regime..."):
        hmm_result = _load_hmm_result()

    with st.spinner(f"Loading {ticker} swing-trading context..."):
        swing_result = build_swing_result(
            ticker=ticker,
            benchmark=benchmark,
            context=context,
            market_regime=str(latest["Regime"]),
            swing_timeframe=swing_timeframe,
            volatility_timeframe_config=volatility_timeframe_config,
        )

    regime_persistence = analyze_regime_persistence(display_scored)
    latest_rvol = _optional_float(latest.get("RVOL Medium"))
    if latest_rvol is None:
        latest_rvol = _optional_float(latest.get("RVOL 20D"))
    with st.spinner(f"Loading {ticker} options and event overlays..."):
        options_context = _load_options_volatility_context(
            ticker=ticker,
            price_df=market_data.ticker_ohlcv,
            latest_regime=str(latest["Regime"]),
            latest_rvol=latest_rvol,
            finviz_snapshot=ticker_metadata,
        )
        event_context = _load_event_context(
            ticker=ticker,
            latest_rvol=latest_rvol,
            volume_context=str(latest.get("Volume Context", "Unavailable")),
            options_context=options_context.context,
        )

    active_tab = _render_tab_buttons(
        [
            "Overview",
            "Performance Comparison",
            "Peers / Sector / Industry / Theme",
            "Recommendation",
            "Relative Context",
            "Swing Trading",
        ]
    )
    if active_tab == "Overview":
        _render_scope_badges(
            [
                ("Score", "Latest MR-1 reading"),
                ("Score Scope", score_profile["scope"]),
                ("Dashboard TF", timeframe_label),
                ("Sensitivity", sensitivity),
                ("Signal Lookbacks", _lookback_summary(lookbacks)),
            ]
        )
        _render_summary(
            ticker=ticker,
            benchmark=benchmark,
            timeframe_label=timeframe_label,
            score_scope=score_profile["scope"],
            score_description=score_profile["description"],
            swing_timeframe=swing_timeframe,
            swing_scope=swing_profile["scope"],
            sensitivity=sensitivity,
            signal_summary=_lookback_summary(lookbacks),
            core_score=int(latest["MR-1 Core Score"]),
            volume_adjustment=int(latest["Volume Adjustment"]),
            latest_score=int(latest["MR-1 Score"]),
            latest_regime=str(latest["Regime"]),
            exposure=float(latest["Exposure"]),
            positive_driver=positive_driver,
            negative_driver=negative_driver,
        )
        _render_relative_context_card(
            relative_context=relative_context,
            base_score=int(latest["MR-1 Score"]),
            adjusted_score=relative_adjusted_score,
            adjusted_regime=relative_adjusted_regime,
            adjusted_exposure=relative_adjusted_exposure,
        )
        _render_clean_relative_trend_card(clean_relative_trend, show_chart=True)
        _render_regime_score_analysis(scored=scored, signals=signals, latest_regime=str(latest["Regime"]))
        _render_hmm_summary(hmm_result=hmm_result, rule_regime=str(latest["Regime"]))
        _render_regime_guide(str(latest["Regime"]))
        _render_scope_badges([("Reading", "Latest timeframe-aware score"), ("Sensitivity", sensitivity)])
        _render_signal_cards(signals)
        _render_scope_badges(
            [
                (
                    "Volume",
                    f"{volume_timeframe_config['volume_short_window']}D / "
                    f"{volume_timeframe_config['volume_medium_window']}D / "
                    f"{volume_timeframe_config['volume_long_window']}D",
                ),
                ("Volume Preset", volume_preset),
                ("Volume Control", _preset_choice_label(volume_preset_choice)),
                ("Swing Vol Control", _preset_choice_label(volatility_preset_choice)),
                ("Float Turnover", "Finviz avg/current volume"),
                ("Volatility", f"VIX {lookbacks['vix']}D"),
            ]
        )
        _render_volume_context_card(
            latest=latest,
            ticker=ticker,
            ticker_metadata=ticker_metadata,
            context=context,
            regime=str(latest["Regime"]),
            timeframe_config=volume_timeframe_config,
        )
        _render_scope_badges(
            [
                ("Options", "Multiple expirations"),
                ("News", "24h / 7D"),
                ("Regime Persistence", timeframe_label),
                ("Volume Preset", volume_preset),
                ("Swing Vol Preset", volatility_preset),
                ("Event Risk", "7D / earnings proximity"),
            ]
        )
        _render_v2_context_cards(
            regime_persistence=regime_persistence,
            options_context=options_context,
            event_context=event_context,
        )
        _render_scope_badges([("Dashboard TF", timeframe_label), ("Charts", "Filtered display"), ("Score Scope", score_profile["scope"])])
        _render_charts(
            display_scored,
            ticker=ticker,
            benchmark=benchmark,
            options_context=options_context,
            regime_persistence=regime_persistence,
        )

    elif active_tab == "Performance Comparison":
        _render_scope_badges([("Dashboard TF", timeframe_label), ("Backtest", "Filtered display"), ("Score Scope", score_profile["scope"])])
        _render_backtest(display_scored, ticker=ticker)
        _render_mr1_regime_score_comparison(
            ticker=ticker,
            benchmark=benchmark,
            context=context,
            scored=scored,
            timeframe_label=timeframe_label,
            sensitivity=sensitivity,
            lookbacks=lookbacks,
            volume_timeframe_config=volume_timeframe_config,
            key_prefix="performance",
        )
        _render_scope_badges([("Swing Windows", "5D / 10D / 1M / 2M / 3M / 4M / 6M / 8M / 10M / YTD / 1Y")])
        _render_performance_table(swing_result.performance_table, title="Swing Window Performance")

    elif active_tab == "Peers / Sector / Industry / Theme":
        _render_scope_badges(
            [
                ("Dashboard TF", timeframe_label),
                ("Swing TF", swing_timeframe),
                ("Score Scope", score_profile["scope"]),
                ("Volume", "Avg daily / current"),
                ("Volatility", f"VIX {lookbacks['vix']}D"),
            ]
        )
        _render_peer_context(
            ticker=ticker,
            benchmark=benchmark,
            swing_result=swing_result,
            scored=scored,
            ticker_metadata=ticker_metadata,
            context=context,
            timeframe_label=timeframe_label,
            swing_timeframe=swing_timeframe,
            sensitivity=sensitivity,
            lookbacks=lookbacks,
            volume_timeframe_config=volume_timeframe_config,
            options_context=options_context,
            event_context=event_context,
        )

    elif active_tab == "Recommendation":
        _render_scope_badges(
            [
                ("Reading", "Latest only"),
                ("Score Scope", score_profile["scope"]),
                ("Sensitivity", sensitivity),
                ("HMM", "Market regime"),
                ("Options", "Multiple expirations"),
                ("News", "24h / 7D"),
            ]
        )
        _render_recommendation(
            ticker=ticker,
            benchmark=benchmark,
            scored=scored,
            signals=signals,
            latest_score=int(latest["MR-1 Score"]),
            latest_regime=str(latest["Regime"]),
            exposure=float(latest["Exposure"]),
            hmm_result=hmm_result,
            regime_persistence=regime_persistence,
            options_context=options_context,
            event_context=event_context,
            swing_result=swing_result,
            clean_relative_trend=clean_relative_trend,
        )

    elif active_tab == "Relative Context":
        _render_scope_badges(
            [
                ("Layer", "Small MR-1 modifier"),
                ("Dashboard TF", timeframe_label),
                ("Benchmark", comparison_benchmark),
                ("Score Impact", _format_signed_points(int(relative_context.get("score_adjustment", 0)))),
            ]
        )
        _render_relative_context_detail(relative_context=relative_context)

    elif active_tab == "Swing Trading":
        _render_scope_badges(
            [
                ("Swing TF", swing_timeframe),
                ("Swing Scope", swing_result.score_scope),
                ("Vol Preset", volatility_preset),
                ("ATR", f"{volatility_timeframe_config['atr_window']}D"),
                ("Realized Vol", f"{volatility_timeframe_config['realized_vol_window']}D"),
                ("Regime Filter", str(latest["Regime"])),
                ("Options/Event", "Risk overlay"),
            ]
        )
        _render_swing_trading(
            ticker=ticker,
            benchmark=benchmark,
            swing_result=swing_result,
            swing_timeframe=swing_timeframe,
            hmm_result=hmm_result,
            regime_persistence=regime_persistence,
            options_context=options_context,
            event_context=event_context,
            clean_relative_trend=clean_relative_trend,
        )


@st.cache_data(ttl=3600)
def _load_hmm_result() -> HMMResult:
    return build_hmm_result()


@st.cache_data(ttl=1800, show_spinner=False)
def _load_benchmark_ohlcv(benchmark: str, period: str = "1y") -> pd.DataFrame:
    return get_benchmark_ohlcv(benchmark=benchmark, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_options_volatility_context(
    ticker: str,
    price_df: pd.DataFrame,
    latest_regime: str,
    latest_rvol: float | None,
    max_expirations: int = 6,
    finviz_snapshot: dict | None = None,
) -> OptionsVolatilityResult:
    return build_options_volatility_context(
        ticker=ticker,
        price_df=price_df,
        latest_regime=latest_regime,
        latest_rvol=latest_rvol,
        max_expirations=max_expirations,
        finviz_snapshot=finviz_snapshot,
    )


@st.cache_data(ttl=900, show_spinner=False)
def _load_event_context(
    ticker: str,
    latest_rvol: float | None,
    volume_context: str,
    options_context: str,
) -> EventContextResult:
    return build_event_context(
        ticker=ticker,
        latest_rvol=latest_rvol,
        volume_context=volume_context,
        options_context=options_context,
    )


def _render_top_bar() -> tuple[str, st.delta_generator.DeltaGenerator]:
    container = st.container()
    with container:
        col1, col2, col3 = st.columns([1.1, 1.7, 1.3], vertical_alignment="center")
        with col1:
            st.markdown('<div class="app-name">MR-1 Lite</div>', unsafe_allow_html=True)
        with col2:
            ticker = normalize_ticker(
                st.text_input(
                    "Search ticker",
                    value=DEFAULT_TICKER,
                    key="ticker_search",
                    placeholder="AAPL, NVDA, TSLA, QQQ, SPY, BTC-USD, ETH-USD",
                )
            )
        with col3:
            last_updated_slot = st.empty()

    return ticker or DEFAULT_TICKER, last_updated_slot


def _render_last_updated(
    slot: st.delta_generator.DeltaGenerator,
    last_updated: pd.Timestamp | None,
) -> None:
    date_text = last_updated.strftime("%Y-%m-%d") if last_updated is not None else "Unavailable"
    slot.markdown(
        f'<div class="top-date">Last updated: <strong>{date_text}</strong></div>',
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[str, str, str, str, str, str, str, list[str]]:
    with st.sidebar:
        st.header("Controls")
        if "active_benchmark" not in st.session_state:
            st.session_state["active_benchmark"] = DEFAULT_BENCHMARK
        if "benchmark_search" not in st.session_state:
            st.session_state["benchmark_search"] = st.session_state["active_benchmark"]

        with st.form("benchmark_form"):
            st.text_input(
                "Benchmark ticker",
                key="benchmark_search",
                placeholder="SPY, QQQ, IWM, BTC-USD, ETH-USD...",
            )
            benchmark_submitted = st.form_submit_button("Apply benchmark")
            if benchmark_submitted:
                st.session_state["active_benchmark"] = (
                    normalize_ticker(st.session_state["benchmark_search"]) or DEFAULT_BENCHMARK
                )
        benchmark = st.session_state["active_benchmark"]
        st.caption(
            f"Active benchmark: {benchmark}. Crypto slash formats like BTC/USD are converted to BTC-USD."
        )
        timeframe_label = st.selectbox(
            "Timeframe",
            list(TIMEFRAMES.keys()),
            index=list(TIMEFRAMES.keys()).index("1M"),
        )
        swing_timeframe = st.selectbox(
            "Swing timeframe",
            SWING_TIMEFRAMES,
            index=SWING_TIMEFRAMES.index("1M"),
        )
        sensitivity = st.select_slider(
            "Signal sensitivity",
            options=list(SENSITIVITY_LOOKBACKS.keys()),
            value="Balanced",
        )
        preset_options = list(TIMEFRAME_PRESETS.keys())
        volume_preset_choice = st.selectbox(
            "Volume Context preset",
            [AUTO_VOLUME_PRESET_OPTION, *preset_options],
            index=0,
            format_func=_preset_option_label,
            help="Auto follows Timeframe. Manual choices override only Volume Context windows.",
        )
        volatility_preset_choice = st.selectbox(
            "Swing Volatility preset",
            [AUTO_VOLATILITY_PRESET_OPTION, *preset_options],
            index=0,
            format_func=_preset_option_label,
            help="Auto follows Swing timeframe. Manual choices override only ATR and realized-volatility windows.",
        )
        volume_preset, volatility_preset = _resolve_volume_volatility_presets(
            timeframe_label=timeframe_label,
            swing_timeframe=swing_timeframe,
            volume_preset_choice=volume_preset_choice,
            volatility_preset_choice=volatility_preset_choice,
        )
        st.caption(f"Volume Context: {_preset_scope_summary(volume_preset)} via {_preset_choice_label(volume_preset_choice)}.")
        st.caption(f"Swing Volatility: {_volatility_scope_summary(volatility_preset)} via {_preset_choice_label(volatility_preset_choice)}.")
        with st.expander("Peer Override", expanded=False):
            peer_override_mode = st.radio(
                "Peer override mode",
                ["Auto", "Add to auto peers", "Replace auto peers"],
                index=0,
                help="Use Auto for curated/Yahoo peers. Add appends your tickers. Replace uses only your tickers.",
            )
            peer_override_text = st.text_input(
                "Peer tickers",
                value="",
                placeholder="AMD, AVGO, MU, MRVL",
                help="Comma or space separated. Crypto slash formats like BTC/USD are converted to BTC-USD.",
            )
            peer_override_tickers = _parse_ticker_list(peer_override_text)
            if peer_override_mode == "Auto":
                st.caption("Using the dashboard's curated/Yahoo peer logic.")
            elif peer_override_tickers:
                st.caption(f"{peer_override_mode}: {', '.join(peer_override_tickers)}")
            else:
                st.caption("Enter at least one ticker to activate the override.")
        st.markdown("**Timeframe Scope**")
        st.caption("Timeframe: changes MR-1 score horizon, charts, backtests, and Volume Context when its preset is Auto.")
        st.caption("Swing timeframe: changes Swing Score horizon, focused return window, and Swing Volatility when its preset is Auto.")
        st.caption("Signal sensitivity: makes MR-1 slower/smoother or faster/noisier inside the selected Timeframe.")
        st.caption(f"Volume Context preset: Auto follows Timeframe; manual choices affect volume only. Selected averages: {_preset_scope_summary(volume_preset)}.")
        st.caption(f"Swing Volatility preset: Auto follows Swing timeframe; manual choices affect ATR/realized volatility only. Selected averages: {_volatility_scope_summary(volatility_preset)}.")
        st.caption("Shorter settings react faster with more noise; longer settings are smoother but slower.")
        if DEBUG_MODE or VOLUME_CONTEXT_CONFIG.get("show_advanced_controls", False):
            with st.expander("Advanced timeframe details"):
                volume_preset, volatility_preset = _resolve_volume_volatility_presets(
                    timeframe_label=timeframe_label,
                    swing_timeframe=swing_timeframe,
                    volume_preset_choice=volume_preset_choice,
                    volatility_preset_choice=volatility_preset_choice,
                )
                st.markdown("Volume Context")
                st.json(get_timeframe_config(volume_preset))
                st.markdown("Swing Volatility")
                st.json(get_timeframe_config(volatility_preset))
        st.caption("Ticker search lives in the top bar.")

    return (
        benchmark or DEFAULT_BENCHMARK,
        timeframe_label,
        swing_timeframe,
        sensitivity,
        volume_preset_choice,
        volatility_preset_choice,
        peer_override_mode,
        peer_override_tickers,
    )


def _parse_ticker_list(raw: str) -> list[str]:
    tickers: list[str] = []
    for token in str(raw or "").replace(",", " ").split():
        symbol = normalize_ticker(token)
        if symbol and symbol not in tickers:
            tickers.append(symbol)
    return tickers


def _apply_peer_override(context, ticker: str, mode: str, override_tickers: list[str]):
    if mode == "Auto" or not override_tickers:
        return context

    normalized_ticker = normalize_ticker(ticker)
    clean_override = [symbol for symbol in override_tickers if symbol and symbol != normalized_ticker]
    if not clean_override:
        return context

    if mode == "Replace auto peers":
        peers = clean_override
        source = "Manual peer override"
    else:
        peers = []
        for symbol in [*context.peers, *clean_override]:
            if symbol and symbol != normalized_ticker and symbol not in peers:
                peers.append(symbol)
        source = f"{context.peer_source} + manual peer override"

    return replace(context, peers=peers, peer_source=source)


def _resolve_volume_volatility_presets(
    timeframe_label: str,
    swing_timeframe: str,
    volume_preset_choice: str,
    volatility_preset_choice: str,
) -> tuple[str, str]:
    volume_preset = (
        volume_preset_choice
        if volume_preset_choice in TIMEFRAME_PRESETS
        else TIMEFRAME_TO_VOLUME_PRESET.get(timeframe_label, DEFAULT_TIMEFRAME_PRESET)
    )
    volatility_preset = (
        volatility_preset_choice
        if volatility_preset_choice in TIMEFRAME_PRESETS
        else SWING_TIMEFRAME_TO_VOLATILITY_PRESET.get(swing_timeframe, DEFAULT_TIMEFRAME_PRESET)
    )
    return volume_preset, volatility_preset


def _preset_choice_label(choice: str) -> str:
    return "Auto" if choice.startswith("Auto") else "Manual"


def _preset_option_label(choice: str) -> str:
    if choice.startswith("Auto"):
        return choice
    return f"{choice} - {_preset_scope_summary(choice)}; {_volatility_scope_summary(choice)}"


def _preset_scope_summary(preset: str) -> str:
    config = get_timeframe_config(preset)
    percentile_window = int(config["volume_percentile_window"])
    percentile_label = "1Y" if percentile_window >= 252 else f"{percentile_window}D"
    return (
        f"RVOL {config['volume_short_window']}/{config['volume_medium_window']}/{config['volume_long_window']}D, "
        f"percentile {percentile_label}"
    )


def _volatility_scope_summary(preset: str) -> str:
    config = get_timeframe_config(preset)
    return f"ATR {config['atr_window']}D, realized vol {config['realized_vol_window']}D"


def _render_scope_badges(items: list[tuple[str, str]]) -> None:
    badges = "".join(
        f'<span class="scope-badge"><strong>{escape(str(label))}</strong>: {escape(str(value))}</span>'
        for label, value in items
    )
    st.markdown(f'<div class="scope-badges">{badges}</div>', unsafe_allow_html=True)


def _render_tab_buttons(labels: list[str]) -> str:
    if "dashboard_tab_selector" not in st.session_state:
        st.session_state["dashboard_tab_selector"] = labels[0]

    return st.radio(
        "Dashboard section",
        labels,
        horizontal=True,
        key="dashboard_tab_selector",
        label_visibility="collapsed",
    )


def _lookback_summary(lookbacks: dict[str, int]) -> str:
    return (
        f"Trend {lookbacks['trend']}D | VIX {lookbacks['vix']}D | "
        f"RS {lookbacks['relative_strength']}D"
    )


def _render_summary(
    ticker: str,
    benchmark: str,
    timeframe_label: str,
    score_scope: str,
    score_description: str,
    swing_timeframe: str,
    swing_scope: str,
    sensitivity: str,
    signal_summary: str,
    core_score: int,
    volume_adjustment: int,
    latest_score: int,
    latest_regime: str,
    exposure: float,
    positive_driver: str,
    negative_driver: str,
) -> None:
    st.subheader(f"Ticker: {ticker}")
    st.caption(
        f"Latest MR-1 score is timeframe-aware: {score_scope} scope from Dashboard TF {timeframe_label}, "
        f"{sensitivity} sensitivity, and signals {signal_summary}. {score_description} "
        f"Swing TF {swing_timeframe} uses a {swing_scope} swing-scoring horizon."
    )
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("MR-1 Core Score", f"{core_score} / 85")
    col2.metric("Volume Adjustment", f"{volume_adjustment:+d}")
    col3.metric("Final MR-1 Score", f"{latest_score} / 100")
    col4.metric("Regime", latest_regime)
    col5.metric("Suggested Exposure", _format_pct(exposure))
    col6.metric("Active Benchmark", benchmark)

    col4, col5 = st.columns(2)
    col4.info(f"Positive Driver: {positive_driver}")
    col5.warning(f"Warning: {negative_driver}")


def _render_regime_guide(active_regime: str) -> None:
    st.subheader("Regime Guide")
    st.caption("How to interpret the current market state and which strategies usually fit each regime.")

    regimes = {
        "Risk-On": {
            "score": "75-100",
            "meaning": "Market conditions are broadly supportive. Trend, volatility, and participation are favorable enough to take normal risk.",
            "best_strategies": [
                "Trend following and breakout continuation",
                "Buying pullbacks to rising moving averages",
                "Relative strength leaders and momentum swings",
                "Normal position sizing with standard risk controls",
            ],
            "avoid": "Avoid over-leveraging or chasing extended moves without a stop.",
        },
        "Neutral": {
            "score": "45-74",
            "meaning": "The market is mixed. Some signals support risk, but confirmation is incomplete or fading.",
            "best_strategies": [
                "Selective swing trades in strong relative-strength names",
                "Smaller entries and partial positions",
                "Range trading near support/resistance",
                "Pairs or hedged exposure when market support is unclear",
            ],
            "avoid": "Avoid broad aggressive exposure and low-quality laggards.",
        },
        "Defensive": {
            "score": "0-44",
            "meaning": "Risk conditions are poor. Trend, volatility, or breadth signals suggest capital preservation matters more than upside capture.",
            "best_strategies": [
                "Cash preservation and reduced equity exposure",
                "Hedges, inverse exposure, or defensive sectors",
                "Shorter holding periods and tighter stops",
                "Waitlist strong names instead of forcing entries",
            ],
            "avoid": "Avoid new long swing entries unless they are exceptional and tightly managed.",
        },
    }

    cols = st.columns(3)
    for col, (regime, details) in zip(cols, regimes.items()):
        active_class = " active" if regime == active_regime else ""
        strategies = "".join(f"<li>{strategy}</li>" for strategy in details["best_strategies"])
        with col:
            st.markdown(
                f"""
                <div class="regime-card{active_class}">
                    <div class="regime-card-title">{regime}</div>
                    <div class="regime-score">Score: {details["score"]}</div>
                    <p>{details["meaning"]}</p>
                    <div class="regime-section-title">Best strategies</div>
                    <ul>{strategies}</ul>
                    <div class="regime-section-title">Watch out</div>
                    <p>{details["avoid"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_signal_cards(signals) -> None:
    st.subheader("Signal Cards")
    cols = st.columns(len(signals))
    for col, signal in zip(cols, signals):
        with col:
            status_class = signal.status.lower()
            st.markdown(
                f"""
                <div class="signal-card signal-{status_class}">
                    <div class="signal-title">{signal.name}</div>
                    <div class="signal-status {status_class}">{signal.status}</div>
                    <div class="signal-score">{signal.score} / {signal.max_score}</div>
                    <div class="signal-row">Current: {signal.current_value:,.2f}</div>
                    <div class="signal-row">Threshold: {signal.threshold:,.2f}</div>
                    <p>{signal.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_volume_context_card(
    latest: pd.Series,
    ticker: str,
    ticker_metadata: dict,
    context,
    regime: str,
    timeframe_config: dict,
) -> None:
    st.subheader("Volume Context")
    status_class = _volume_status_class(str(latest.get("Volume Context", "Unavailable")))
    adjustment = int(latest.get("Volume Adjustment", 0))
    preset = str(latest.get("Volume Timeframe Preset", timeframe_config.get("preset", DEFAULT_TIMEFRAME_PRESET)))
    short_window = int(latest.get("Volume Short Window", timeframe_config["volume_short_window"]))
    medium_window = int(latest.get("Volume Medium Window", timeframe_config["volume_medium_window"]))
    long_window = int(latest.get("Volume Long Window", timeframe_config["volume_long_window"]))
    percentile_window = int(latest.get("Volume Percentile Window", timeframe_config["volume_percentile_window"]))
    percentile_label = "1Y" if percentile_window >= 252 else f"{percentile_window}D"
    float_turnover_labels = _float_turnover_summary(latest, timeframe_config)
    finviz_available = ticker_metadata.get("source") == "finviz" and bool(ticker_metadata.get("available"))
    finviz_status = "Available" if finviz_available else "Unavailable"
    finviz_note = "" if finviz_available else "Finviz data unavailable. Float turnover skipped unless another source provides shares float."
    volume_table = _build_volume_comparison_table(ticker=ticker, ticker_metadata=ticker_metadata, context=context)
    avg_daily_float_turnover = _metadata_average_float_turnover(ticker_metadata)
    regime_read = _volume_regime_read(
        latest=latest,
        ticker=ticker,
        regime=regime,
        volume_table=volume_table,
    )
    trend_badge = _volume_trend_badge(latest)
    debug_note = ""
    if DEBUG_MODE:
        debug_parts = []
        if ticker_metadata.get("finviz_error"):
            debug_parts.append(f"Finviz warning: {ticker_metadata.get('finviz_error')}")
        preview_url = (ticker_metadata.get("debug") or {}).get("preview_url")
        if preview_url:
            debug_parts.append(f"Finviz preview URL: {preview_url}")
        if debug_parts:
            debug_note = f'<div class="volume-note">{escape(" | ".join(str(part) for part in debug_parts))}</div>'
    finviz_note_html = f'<div class="volume-note">{escape(finviz_note)}</div>' if finviz_note else ""
    st.markdown(
        f"""
        <div class="volume-card signal-{status_class}">
            <div class="volume-header">
                <div>
                    <div class="volume-eyebrow">Volume Context</div>
                    <div class="volume-title">{escape(str(latest.get("Volume Context", "Unavailable")))}</div>
                    <div class="volume-subtitle">{escape(str(latest.get("Volume Status", "Volume unavailable")))} | {preset} | {trend_badge}</div>
                </div>
                <div class="volume-adjustment {status_class}">{adjustment:+d}</div>
            </div>
            <div class="volume-metric-grid">
                <div class="volume-metric">
                    <span>RVOL {short_window}D</span>
                    <strong>{_format_optional_multiple(latest.get("RVOL Short"))}</strong>
                    <small>short window</small>
                </div>
                <div class="volume-metric">
                    <span>RVOL {medium_window}D</span>
                    <strong>{_format_optional_multiple(latest.get("RVOL Medium"))}</strong>
                    <small>medium confirmation</small>
                </div>
                <div class="volume-metric">
                    <span>RVOL {long_window}D</span>
                    <strong>{_format_optional_multiple(latest.get("RVOL Long"))}</strong>
                    <small>long confirmation</small>
                </div>
                <div class="volume-metric">
                    <span>Volume Percentile</span>
                    <strong>{_format_optional_percentile(latest.get("Volume Percentile"))}</strong>
                    <small>{percentile_label} selected window</small>
                </div>
                <div class="volume-metric">
                    <span>Float Turnover</span>
                    <strong>{_format_optional_pct(latest.get("Daily Float Turnover"))}</strong>
                    <small>{escape(float_turnover_labels)}</small>
                </div>
                <div class="volume-metric">
                    <span>Finviz RVOL</span>
                    <strong>{_format_optional_multiple(ticker_metadata.get("relative_volume"))}</strong>
                    <small>Finviz-provided average/current volume</small>
                </div>
            </div>
            <div class="volume-analysis">
                <p>{escape(str(latest.get("Volume Explanation", "Volume context is unavailable.")))}</p>
                <p>{escape(regime_read)}</p>
            </div>
            <div class="volume-foot">
                <span>Finviz: {finviz_status}</span>
                <span>Float: {_format_optional_number(ticker_metadata.get("shares_float"))}</span>
                <span>Current float turnover: {_format_optional_pct(latest.get("Daily Float Turnover"))}</span>
                <span>Float %: {_format_optional_pct(ticker_metadata.get("float_percent"))}</span>
                <span>{escape(float_turnover_labels)}</span>
                <span>Trades: {_format_optional_number(ticker_metadata.get("trades"))}</span>
                <span>RSI: {_format_optional_decimal(ticker_metadata.get("rsi"))}</span>
                <span>Short ratio: {_format_optional_decimal(ticker_metadata.get("short_ratio"))}</span>
                <span>Earnings: {escape(str(ticker_metadata.get("earnings_date") or "N/A"))}</span>
                <span>{escape(str(ticker_metadata.get("sector") or "Sector unavailable"))}</span>
                <span>{escape(str(ticker_metadata.get("industry") or "Industry unavailable"))}</span>
                <span>{escape(str(ticker_metadata.get("country") or "Country unavailable"))}</span>
            </div>
            {debug_note}
            {finviz_note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not volume_table.empty:
        st.markdown("**Ticker vs Peers / Sector Volume**")
        st.dataframe(
            _style_volume_comparison_table(volume_table),
            use_container_width=True,
            hide_index=True,
            height=_table_height(volume_table),
        )
        st.plotly_chart(
            _average_float_turnover_chart(volume_table),
            width="stretch",
            key=f"overview_avg_float_turnover_{ticker}",
        )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_volume_metadata(symbols: tuple[str, ...]) -> dict[str, dict]:
    metadata = {}
    for symbol in symbols:
        try:
            metadata[symbol] = get_ticker_metadata(symbol)
        except Exception as exc:
            metadata[symbol] = {"ticker": symbol, "source": "unavailable", "error": str(exc)}
    return metadata


def _build_volume_comparison_table(ticker: str, ticker_metadata: dict, context) -> pd.DataFrame:
    assets = [
        ("Selected ticker", ticker.upper(), "Ticker"),
        ("Sector ETF", context.sector_etf, "Sector ETF"),
        ("Industry proxy", context.industry_proxy, "Industry Proxy"),
        ("Theme proxy", context.theme_ticker, "Theme Proxy"),
        ("Sub-industry proxy", context.sub_industry_ticker, "Sub-Industry Proxy"),
    ]
    assets.extend((peer, peer, "Peer") for peer in context.peers)

    deduped_assets = []
    seen = set()
    for label, symbol, asset_type in assets:
        normalized = str(symbol or "").upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_assets.append((label, normalized, asset_type))

    symbols = tuple(symbol for _, symbol, _ in deduped_assets)
    metadata_map = _load_volume_metadata(symbols)
    if ticker.upper() in metadata_map:
        metadata_map[ticker.upper()] = {**metadata_map[ticker.upper()], **ticker_metadata}

    rows = []
    for label, symbol, asset_type in deduped_assets:
        metadata = metadata_map.get(symbol, {})
        volume = _optional_float(metadata.get("volume"))
        average_volume = _optional_float(metadata.get("average_volume"))
        shares_float = _optional_float(metadata.get("shares_float"))
        current_float_turnover = volume / shares_float if volume is not None and shares_float else None
        average_float_turnover = average_volume / shares_float if average_volume is not None and shares_float else None
        rows.append(
            {
                "Asset": label,
                "Ticker": symbol,
                "Type": asset_type,
                "Finviz RVOL": _optional_float(metadata.get("relative_volume")),
                "Volume": volume,
                "Avg Volume": average_volume,
                "Float": shares_float,
                "Current Float Turnover": current_float_turnover,
                "Avg Daily Float Turnover": average_float_turnover,
                "Float %": _optional_float(metadata.get("float_percent")),
                "Short Float": _optional_float(metadata.get("short_float")),
                "Short Interest": _optional_float(metadata.get("short_interest")),
                "Beta": _optional_float(metadata.get("beta")),
                "Gap": _optional_float(metadata.get("gap")),
                "Change From Open": _optional_float(metadata.get("change_from_open")),
                "Finviz Volatility Week": _optional_float(metadata.get("volatility_week")),
                "Finviz Volatility Month": _optional_float(metadata.get("volatility_month")),
                "Trades": _optional_float(metadata.get("trades")),
                "Source": metadata.get("source") or "unavailable",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    peer_rvol = table.loc[table["Type"] == "Peer", "Finviz RVOL"].dropna()
    peer_median = float(peer_rvol.median()) if not peer_rvol.empty else None
    sector_rows = table.loc[table["Type"] == "Sector ETF", "Finviz RVOL"].dropna()
    sector_rvol = float(sector_rows.iloc[0]) if not sector_rows.empty else None

    table["RVOL vs Peer Median"] = table["Finviz RVOL"].map(
        lambda value: _relative_gap(value, peer_median)
    )
    table["RVOL vs Sector"] = table["Finviz RVOL"].map(lambda value: _relative_gap(value, sector_rvol))
    return table


def _style_volume_comparison_table(table: pd.DataFrame):
    display = _apply_display_type(table)[
        [
            "Type",
            "Ticker",
            "Finviz RVOL",
            "RVOL vs Peer Median",
            "RVOL vs Sector",
            "Volume",
            "Avg Volume",
            "Gap",
            "Change From Open",
            "Avg Daily Float Turnover",
            "Current Float Turnover",
            "Float %",
            "Short Float",
            "Finviz Volatility Month",
            "Beta",
            "Source",
        ]
    ].copy()
    styled = (
        display.style.format(
            {
                "Finviz RVOL": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}x",
                "RVOL vs Peer Median": lambda value: "N/A" if pd.isna(value) else f"{value:+.1%}",
                "RVOL vs Sector": lambda value: "N/A" if pd.isna(value) else f"{value:+.1%}",
                "Volume": lambda value: "N/A" if pd.isna(value) else _format_optional_number(value),
                "Avg Volume": lambda value: "N/A" if pd.isna(value) else _format_optional_number(value),
                "Gap": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Change From Open": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Avg Daily Float Turnover": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Current Float Turnover": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Float %": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Short Float": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Finviz Volatility Month": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Beta": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
            },
            na_rep="N/A",
        )
        .map(_rvol_style, subset=["Finviz RVOL"])
        .map(_gap_style, subset=["RVOL vs Peer Median", "RVOL vs Sector"])
        .map(_performance_pct_style, subset=["Gap", "Change From Open"])
        .set_properties(
            subset=["Ticker", "Type"],
            **{"font-weight": "700", "color": "#e5e7eb"},
        )
    )
    return styled


def _style_peer_persistence_table(table: pd.DataFrame):
    return (
        table.style.format(
            {
                "Score Cushion": lambda value: "N/A" if pd.isna(value) else f"{value:+.0f}",
                "Time Risk-On %": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Stability": lambda value: "N/A" if pd.isna(value) else f"{value:.0f}",
            },
            na_rep="N/A",
        )
        .map(_cushion_style, subset=["Score Cushion"])
        .map(_stability_style, subset=["Stability"])
        .set_properties(subset=["Ticker", "Current Regime"], **{"font-weight": "700", "color": "#e5e7eb"})
    )


def _style_peer_options_table(table: pd.DataFrame):
    return (
        table.style.format(
            {
                "Current IV": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "IV Rank": lambda value: "N/A" if pd.isna(value) else f"{value:.0%}",
                "IV Percentile": lambda value: "N/A" if pd.isna(value) else f"{value:.0%}",
                "Finviz Vol W": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "Finviz Vol M": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "IV Premium": lambda value: "N/A" if pd.isna(value) else f"{value:+.1%}",
                "Put/Call Vol": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
                "Put/Call OI": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
                "DTE": lambda value: "N/A" if pd.isna(value) else f"{value:.0f}",
                "Expirations": lambda value: "N/A" if pd.isna(value) else f"{value:.0f}",
            },
            na_rep="N/A",
        )
        .map(_iv_style, subset=["Current IV", "IV Premium"])
        .set_properties(subset=["Ticker", "Context"], **{"font-weight": "700", "color": "#e5e7eb"})
    )


def _style_options_expiration_table(table: pd.DataFrame):
    display_columns = [
        "Expiration",
        "DTE",
        "ATM Blended IV",
        "ATM Call IV",
        "ATM Put IV",
        "IV Source",
        "IV Premium",
        "Put/Call Vol",
        "Put/Call OI",
        "Options Volume",
        "Open Interest",
        "Volume/OI",
    ]
    display = table[[column for column in display_columns if column in table.columns]].copy()
    return (
        display.style.format(
            {
                "DTE": lambda value: "N/A" if pd.isna(value) else f"{value:.0f}",
                "ATM Blended IV": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "ATM Call IV": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "ATM Put IV": lambda value: "N/A" if pd.isna(value) else f"{value:.1%}",
                "IV Premium": lambda value: "N/A" if pd.isna(value) else f"{value:+.1%}",
                "Put/Call Vol": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
                "Put/Call OI": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
                "Options Volume": lambda value: "N/A" if pd.isna(value) else _format_optional_number(value),
                "Open Interest": lambda value: "N/A" if pd.isna(value) else _format_optional_number(value),
                "Volume/OI": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}",
            },
            na_rep="N/A",
        )
        .map(_iv_style, subset=["ATM Blended IV", "IV Premium"])
        .set_properties(subset=["Expiration"], **{"font-weight": "800", "color": "#e5e7eb"})
    )


def _style_peer_events_table(table: pd.DataFrame):
    return (
        table.style.format(
            {
                "RVOL": lambda value: "N/A" if pd.isna(value) else f"{value:.2f}x",
                "Days to Earnings": lambda value: "N/A" if pd.isna(value) else f"{value:.0f}",
            },
            na_rep="N/A",
        )
        .map(_rvol_style, subset=["RVOL"])
        .set_properties(subset=["Ticker", "Event Risk", "Catalyst Status"], **{"font-weight": "700", "color": "#e5e7eb"})
    )


def _relevant_context_symbols(ticker: str, benchmark: str, context) -> list[str]:
    return [symbol for symbol, _, _ in _regime_chart_assets(ticker=ticker, benchmark=benchmark, context=context)]


def _regime_chart_assets(ticker: str, benchmark: str, context) -> list[tuple[str, str, str]]:
    candidates = [
        (ticker, "Ticker", "Ticker"),
        (benchmark, "Benchmark", "Benchmark"),
        (context.sector_etf, "Sector ETF", "Sector"),
        (context.industry_proxy, "Industry Proxy", "Industry"),
        (context.theme_ticker, "Theme Proxy", "Theme"),
        (context.sub_industry_ticker, "Sub-Industry Proxy", "Sub-Industry"),
        *[(peer, "Peer", "Peer") for peer in context.peers],
    ]
    assets = []
    seen = set()
    for candidate, role, group in candidates:
        symbol = str(candidate or "").upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            assets.append((symbol, role, group))
    return assets


def _render_mr1_regime_score_comparison(
    ticker: str,
    benchmark: str,
    context,
    scored: pd.DataFrame,
    timeframe_label: str,
    sensitivity: str,
    lookbacks: dict[str, int],
    volume_timeframe_config: dict,
    key_prefix: str,
) -> None:
    st.subheader("MR-1 Regime Score Comparison")
    assets = _regime_chart_assets(ticker=ticker, benchmark=benchmark, context=context)
    symbols = tuple(symbol for symbol, _, _ in assets)
    with st.spinner("Loading MR-1 score histories for comparison..."):
        regime_histories = _load_peer_regime_histories(
            symbols=symbols,
            benchmark=benchmark.upper(),
            period=TIMEFRAMES[timeframe_label],
            lookback_items=tuple(sorted(lookbacks.items())),
            volume_timeframe_items=_hashable_timeframe_config(volume_timeframe_config),
        )
    if not scored.empty:
        regime_histories[ticker.upper()] = scored
    if not regime_histories:
        st.info("MR-1 score comparison is unavailable for the selected assets.")
        return

    label_by_symbol = {
        symbol: f"{role}: {symbol}" if role != "Peer" else f"Peer: {symbol}"
        for symbol, role, _ in assets
        if symbol in regime_histories
    }
    symbol_by_label = {label: symbol for symbol, label in label_by_symbol.items()}
    default_labels = [
        label
        for symbol, label in label_by_symbol.items()
        if next((group for asset_symbol, _, group in assets if asset_symbol == symbol), "") != "Peer"
    ]
    if not default_labels:
        default_labels = list(symbol_by_label.keys())[:4]

    selected_labels = st.multiselect(
        "Show MR-1 score lines",
        options=list(symbol_by_label.keys()),
        default=default_labels,
        key=f"{key_prefix}_mr1_score_lines",
        help="Toggle ticker, benchmark, sector/industry/theme proxies, and peers on or off.",
    )
    selected_symbols = {symbol_by_label[label] for label in selected_labels if label in symbol_by_label}
    if not selected_symbols:
        st.info("Select at least one line to show the MR-1 score chart.")
        return

    _render_scope_badges([("Dashboard TF", timeframe_label), ("Sensitivity", sensitivity), ("Benchmark", benchmark.upper())])
    st.plotly_chart(
        _peer_regime_score_chart(
            regime_histories=regime_histories,
            ticker=ticker.upper(),
            timeframe_label=timeframe_label,
            assets=assets,
            selected_symbols=selected_symbols,
        ),
        width="stretch",
        key=f"{key_prefix}_mr1_regime_scores_{ticker}_{benchmark}_{timeframe_label}_{sensitivity}",
    )


def _hashable_timeframe_config(timeframe_config: dict) -> tuple[tuple[str, object], ...]:
    items = []
    for key, value in timeframe_config.items():
        if isinstance(value, list):
            items.append((key, tuple(value)))
        else:
            items.append((key, value))
    return tuple(sorted(items))


@st.cache_data(ttl=3600, show_spinner=False)
def _load_peer_regime_histories(
    symbols: tuple[str, ...],
    benchmark: str,
    period: str,
    lookback_items: tuple[tuple[str, int], ...],
    volume_timeframe_items: tuple[tuple[str, object], ...],
) -> dict[str, pd.DataFrame]:
    lookbacks = dict(lookback_items)
    volume_timeframe_config = dict(volume_timeframe_items)
    if "float_turnover_windows" in volume_timeframe_config:
        volume_timeframe_config["float_turnover_windows"] = list(volume_timeframe_config["float_turnover_windows"])
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            market_data = fetch_market_data(ticker=symbol, benchmark=benchmark, period=period)
            clean_close = market_data.close.dropna()
            if clean_close.empty:
                continue
            indicators = calculate_indicators(clean_close, lookbacks=lookbacks)
            if indicators.empty:
                continue
            metadata = get_ticker_metadata(symbol)
            scored = score_history(
                indicators,
                ticker_ohlcv=market_data.ticker_ohlcv,
                shares_float=metadata.get("shares_float"),
                volume_timeframe_config=volume_timeframe_config,
            )
            if not scored.empty:
                histories[symbol] = scored
        except Exception:
            continue
    return histories


def _average_float_turnover_chart(volume_table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    chart_data = volume_table.dropna(subset=["Avg Daily Float Turnover"]).copy()
    if chart_data.empty:
        fig.add_annotation(
            text="Average daily float-turnover is unavailable for these assets.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#cbd5e1", size=13),
        )
        return _finish_chart(fig, title="Average Daily Float Turnover")

    chart_data = chart_data.sort_values("Avg Daily Float Turnover", ascending=False)
    colors = ["#38bdf8" if asset_type == "Ticker" else "#14b8a6" if "Proxy" in asset_type or "ETF" in asset_type else "#64748b" for asset_type in chart_data["Type"]]
    fig.add_trace(
        go.Bar(
            x=chart_data["Ticker"],
            y=chart_data["Avg Daily Float Turnover"],
            name="Avg daily float turnover",
            marker=dict(color=colors),
            hovertemplate="%{x}<br>Avg Daily Float Turnover: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".1%")
    return _finish_chart(fig, title="Average Daily Float Turnover")


def _peer_regime_score_chart(
    regime_histories: dict[str, pd.DataFrame],
    ticker: str,
    timeframe_label: str,
    assets: list[tuple[str, str, str]] | None = None,
    selected_symbols: set[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=75, y1=100, fillcolor="#17803d", opacity=0.16, line_width=0, layer="below")
    fig.add_hrect(y0=45, y1=75, fillcolor="#c99400", opacity=0.16, line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=45, fillcolor="#b83232", opacity=0.16, line_width=0, layer="below")
    fig.add_hline(y=75, line=dict(color="#22c55e", width=1.3, dash="dash"))
    fig.add_hline(y=45, line=dict(color="#f59e0b", width=1.3, dash="dash"))

    asset_lookup = {symbol: (role, group) for symbol, role, group in assets or []}
    selected_symbols = selected_symbols or set(regime_histories)
    for symbol, scored in regime_histories.items():
        if symbol not in selected_symbols:
            continue
        display = _filter_scored_by_timeframe(scored, timeframe_label)
        if display.empty or "MR-1 Score" not in display.columns:
            continue
        is_focus = symbol == ticker
        role, group = asset_lookup.get(symbol, ("Peer", "Peer"))
        fig.add_trace(
            go.Scatter(
                x=display.index,
                y=display["MR-1 Score"],
                name=f"{role}: {symbol}" if role != "Ticker" else f"Ticker: {symbol}",
                mode="lines",
                line=dict(
                    color=_regime_score_line_color(group, is_focus),
                    width=3.4 if is_focus else 1.6,
                ),
                opacity=1.0 if is_focus else 0.68,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y}<extra></extra>",
            )
        )

    fig.update_yaxes(range=[0, 100])
    return _finish_chart(fig, title="MR-1 Regime Scores: Ticker / Proxies / Peers")


def _regime_score_line_color(group: str, is_focus: bool) -> str:
    if is_focus:
        return "#38bdf8"
    colors = {
        "Benchmark": "#f8fafc",
        "Sector": "#22c55e",
        "Industry": "#14b8a6",
        "Theme": "#a78bfa",
        "Sub-Industry": "#f59e0b",
        "Peer": "#94a3b8",
    }
    return colors.get(group, "#94a3b8")


def _build_peer_options_table(regime_histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, scored in list(regime_histories.items())[:10]:
        if scored.empty or "Ticker Close" not in scored.columns:
            continue
        latest = scored.iloc[-1]
        price_df = pd.DataFrame({"Close": scored["Ticker Close"]})
        if "Volume" in scored.columns:
            price_df["Volume"] = scored["Volume"]
        result = _load_options_volatility_context(
            ticker=symbol,
            price_df=price_df,
            latest_regime=str(latest.get("Regime", "Unavailable")),
            latest_rvol=_selected_rvol(latest),
            max_expirations=2,
        )
        rows.append(
            {
                "Ticker": symbol,
                "Current IV": result.current_iv,
                "IV Rank": result.iv_rank,
                "IV Percentile": result.iv_percentile,
                "IV Premium": result.iv_premium,
                "Put/Call Vol": result.put_call_volume_ratio,
                "Put/Call OI": result.put_call_oi_ratio,
                "DTE": result.days_to_expiration,
                "Expirations": result.expirations_analyzed,
                "Context": result.context,
            }
        )
    return pd.DataFrame(rows)


def _build_peer_events_table(regime_histories: dict[str, pd.DataFrame], peer_options: pd.DataFrame) -> pd.DataFrame:
    options_context_by_symbol = {}
    if not peer_options.empty:
        options_context_by_symbol = dict(zip(peer_options["Ticker"], peer_options["Context"]))

    rows = []
    for symbol, scored in list(regime_histories.items())[:10]:
        if scored.empty:
            continue
        latest = scored.iloc[-1]
        result = _load_event_context(
            ticker=symbol,
            latest_rvol=_selected_rvol(latest),
            volume_context=str(latest.get("Volume Context", "Unavailable")),
            options_context=options_context_by_symbol.get(symbol, "Unavailable"),
        )
        rows.append(
            {
                "Ticker": symbol,
                "RVOL": _selected_rvol(latest),
                "News 24h": result.news_count_24h,
                "News 7D": result.news_count_7d,
                "Days to Earnings": result.days_to_earnings,
                "Event Risk": result.event_risk,
                "Catalyst Status": result.catalyst_status,
            }
        )
    return pd.DataFrame(rows)


def _volume_regime_read(latest: pd.Series, ticker: str, regime: str, volume_table: pd.DataFrame) -> str:
    context = str(latest.get("Volume Context", "Unavailable"))
    ticker_row = _volume_row(volume_table, ticker.upper())
    ticker_rvol = _optional_float(ticker_row.get("Finviz RVOL")) if ticker_row is not None else None
    peer_rows = volume_table[volume_table["Type"] == "Peer"] if not volume_table.empty else pd.DataFrame()
    peer_rvol = peer_rows["Finviz RVOL"].dropna() if "Finviz RVOL" in peer_rows else pd.Series(dtype=float)
    peer_median = float(peer_rvol.median()) if not peer_rvol.empty else None
    sector_rows = volume_table[volume_table["Type"] == "Sector ETF"] if not volume_table.empty else pd.DataFrame()
    sector_rvol = _optional_float(sector_rows.iloc[0].get("Finviz RVOL")) if not sector_rows.empty else None
    rank_text = _volume_rank_text(ticker=ticker.upper(), volume_table=volume_table)
    peer_text = _comparison_text(ticker_rvol, peer_median, "peer median")
    sector_text = _comparison_text(ticker_rvol, sector_rvol, "sector ETF")

    if regime == "Risk-On":
        regime_text = (
            "In Risk-On, strong ticker volume is constructive when it is also competitive against peers and sector; "
            "trend continuation and breakout pullback strategies fit best."
        )
        if context in {"Distribution", "Panic / Liquidation"}:
            regime_text = (
                "Risk-On normally supports long exposure, but distribution volume is a size-control warning; "
                "favor tighter stops and avoid adding into heavy down-volume."
            )
    elif regime == "Neutral":
        regime_text = (
            "In Neutral, volume has to do more of the confirmation work. Above-peer RVOL supports selective swing trades; "
            "below-peer RVOL argues for smaller size or waiting for confirmation."
        )
    else:
        regime_text = (
            "In Defensive, high volume is not automatically bullish. Accumulation can support tactical trades, "
            "but heavy down-volume or weak peer participation favors cash, hedges, or short-duration setups."
        )

    return f"{rank_text} {peer_text} {sector_text} {regime_text}"


def _volume_trend_badge(latest: pd.Series) -> str:
    context = str(latest.get("Volume Context", "Unavailable"))
    if context in {"Accumulation", "Breakout Confirmation"}:
        return "participation confirming the move"
    if context in {"Distribution", "Panic / Liquidation"}:
        return "selling pressure is elevated"
    if context == "Weak Participation":
        return "price move lacks sponsorship"
    if context == "Unavailable":
        return "volume data unavailable"
    return "participation is neutral"


def _volume_row(table: pd.DataFrame, ticker: str) -> pd.Series | None:
    if table.empty:
        return None
    rows = table[table["Ticker"] == ticker]
    return None if rows.empty else rows.iloc[0]


def _volume_rank_text(ticker: str, volume_table: pd.DataFrame) -> str:
    if volume_table.empty or "Finviz RVOL" not in volume_table.columns:
        return f"{ticker} has no cross-sectional RVOL rank available."
    ranked = volume_table.dropna(subset=["Finviz RVOL"]).sort_values("Finviz RVOL", ascending=False)
    if ranked.empty or ticker not in set(ranked["Ticker"]):
        return f"{ticker} has no cross-sectional RVOL rank available."
    rank = int(ranked.index.get_loc(ranked[ranked["Ticker"] == ticker].index[0]) + 1)
    return f"{ticker} ranks #{rank} of {len(ranked)} tracked peer/sector assets by Finviz relative volume."


def _comparison_text(value: float | None, baseline: float | None, label: str) -> str:
    gap = _relative_gap(value, baseline)
    if gap is None:
        return f"Comparison vs {label} is unavailable."
    if gap >= 0.15:
        return f"RVOL is {gap:.0%} above the {label}."
    if gap <= -0.15:
        return f"RVOL is {abs(gap):.0%} below the {label}."
    return f"RVOL is broadly in line with the {label}."


def _relative_gap(value, baseline) -> float | None:
    value = _optional_float(value)
    baseline = _optional_float(baseline)
    if value is None or baseline in (None, 0):
        return None
    return value / baseline - 1


def _metadata_average_float_turnover(metadata: dict) -> float | None:
    average_volume = _optional_float(metadata.get("average_volume"))
    shares_float = _optional_float(metadata.get("shares_float"))
    if average_volume is None or not shares_float:
        return None
    return average_volume / shares_float


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_asset_type(asset_type: str, ticker: str) -> str:
    symbol = str(ticker or "").upper()
    if _is_known_etf_symbol(symbol):
        return "ETF"
    return str(asset_type or "N/A")


def _is_known_etf_symbol(symbol: str) -> bool:
    normalized = str(symbol or "").upper()
    if not normalized or normalized.endswith("-USD"):
        return False
    return normalized in {symbol.upper() for symbol in KNOWN_ETF_SYMBOLS if symbol}


def _apply_display_type(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if "Type" in display.columns and "Ticker" in display.columns:
        display["Type"] = display.apply(lambda row: _display_asset_type(row["Type"], row["Ticker"]), axis=1)
    return display


def _rvol_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value >= 1.5:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.22); font-weight: 800;"
    if value <= 0.75:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.22); font-weight: 800;"
    return "color: #e5e7eb; background-color: rgba(148, 163, 184, 0.10); font-weight: 700;"


def _cushion_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value >= 10:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.22); font-weight: 800;"
    if value <= 3:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.22); font-weight: 800;"
    return "color: #fde68a; background-color: rgba(245, 158, 11, 0.18); font-weight: 700;"


def _stability_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value >= 70:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.22); font-weight: 800;"
    if value < 45:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.22); font-weight: 800;"
    return "color: #fde68a; background-color: rgba(245, 158, 11, 0.18); font-weight: 700;"


def _iv_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value >= 0.35:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.22); font-weight: 800;"
    if value <= 0.0:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.18); font-weight: 700;"
    return "color: #fde68a; background-color: rgba(245, 158, 11, 0.16); font-weight: 700;"


def _overlay_action_style(value) -> str:
    text = str(value).lower()
    if any(word in text for word in ["stress", "event", "unexplained", "fragile", "reduce", "avoid"]):
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.20); font-weight: 800;"
    if any(word in text for word in ["risk-on", "stable", "confirmed", "normal", "buy"]):
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.18); font-weight: 800;"
    return "color: #e5e7eb; background-color: rgba(148, 163, 184, 0.10); font-weight: 700;"


def _gap_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value >= 0.15:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.20); font-weight: 800;"
    if value <= -0.15:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.22); font-weight: 800;"
    return "color: #fde68a; background-color: rgba(245, 158, 11, 0.14); font-weight: 700;"


def _render_regime_score_analysis(scored: pd.DataFrame, signals, latest_regime: str) -> None:
    st.subheader("MR-1 Score Analysis")
    latest_score = int(scored.iloc[-1]["MR-1 Score"])
    score_delta_5d = _score_delta(scored, 5)
    score_delta_21d = _score_delta(scored, 21)
    regime_days = _days_in_current_regime(scored)
    position_label, position_value = _regime_score_position(latest_score=latest_score, latest_regime=latest_regime)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score Momentum 5D", _format_signed_points(score_delta_5d), delta=_format_signed_points(score_delta_5d))
    col2.metric("Score Momentum 1M", _format_signed_points(score_delta_21d), delta=_format_signed_points(score_delta_21d))
    col3.metric("Current Regime Duration", f"{regime_days} trading days")
    col4.metric(position_label, f"{position_value} pts")

    component_table = _regime_component_table(signals)
    st.dataframe(
        _style_regime_component_table(component_table),
        use_container_width=True,
        hide_index=True,
        height=_table_height(component_table),
    )

    breadth_read = _breadth_leadership_read(scored.iloc[-1])
    st.info(_regime_analysis_summary(latest_score=latest_score, latest_regime=latest_regime, signals=signals, breadth_read=breadth_read))


def _regime_component_table(signals) -> pd.DataFrame:
    rows = []
    for signal in signals:
        missing_points = signal.max_score - signal.score
        contribution = signal.score / signal.max_score if signal.max_score else 0.0
        rows.append(
            {
                "Component": signal.name,
                "Status": signal.status,
                "Score": signal.score,
                "Max": signal.max_score,
                "Contribution": contribution,
                "Missing Points": missing_points,
                "Current": signal.current_value,
                "Threshold": signal.threshold,
                "Read": _component_read(signal),
            }
        )
    return pd.DataFrame(rows)


def _style_regime_component_table(table: pd.DataFrame):
    return (
        table.style.format(
            {
                "Contribution": _format_pct,
                "Current": "{:,.2f}",
                "Threshold": "{:,.2f}",
            }
        )
        .map(_component_status_style, subset=["Status"])
        .map(_component_contribution_style, subset=["Contribution"])
        .map(_missing_points_style, subset=["Missing Points"])
        .set_properties(
            subset=["Component", "Read"],
            **{
                "color": "#e5e7eb",
                "font-weight": "700",
            },
        )
    )


def _component_read(signal) -> str:
    if signal.score == signal.max_score:
        return "Full support"
    if signal.score > 0:
        return "Partial support"
    if signal.name == "Volatility":
        return "Volatility pressure"
    if signal.name == "Market Breadth / Leadership":
        return "Breadth or leadership missing"
    return "No support"


def _component_status_style(value: str) -> str:
    if value == "Positive":
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.24); font-weight: 800;"
    if value == "Mixed":
        return "color: #fde68a; background-color: rgba(245, 158, 11, 0.22); font-weight: 800;"
    return "color: #fecaca; background-color: rgba(220, 38, 38, 0.24); font-weight: 800;"


def _component_contribution_style(value: float) -> str:
    if pd.isna(value):
        return ""
    if value >= 0.75:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.20); font-weight: 800;"
    if value > 0:
        return "color: #fde68a; background-color: rgba(245, 158, 11, 0.18); font-weight: 800;"
    return "color: #fecaca; background-color: rgba(220, 38, 38, 0.20); font-weight: 800;"


def _missing_points_style(value: float) -> str:
    if pd.isna(value):
        return ""
    if value == 0:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.16); font-weight: 700;"
    if value <= 5:
        return "color: #fde68a; background-color: rgba(245, 158, 11, 0.14); font-weight: 700;"
    return "color: #fecaca; background-color: rgba(220, 38, 38, 0.18); font-weight: 700;"


def _score_delta(scored: pd.DataFrame, days: int) -> int:
    if len(scored.index) <= days:
        return 0
    return int(scored.iloc[-1]["MR-1 Score"] - scored.iloc[-days - 1]["MR-1 Score"])


def _days_in_current_regime(scored: pd.DataFrame) -> int:
    if scored.empty:
        return 0
    current_regime = scored.iloc[-1]["Regime"]
    count = 0
    for regime in reversed(scored["Regime"].tolist()):
        if regime != current_regime:
            break
        count += 1
    return count


def _regime_score_position(latest_score: int, latest_regime: str) -> tuple[str, int]:
    if latest_regime == "Risk-On":
        return "Risk-On Cushion", latest_score - REGIME_RULES["Risk-On"]["min_score"]
    if latest_regime == "Neutral":
        return "Pts To Risk-On", REGIME_RULES["Risk-On"]["min_score"] - latest_score
    return "Pts To Neutral", REGIME_RULES["Neutral"]["min_score"] - latest_score


def _format_signed_points(value: int) -> str:
    return f"{value:+d} pts"


def _breadth_leadership_read(latest: pd.Series) -> str:
    breadth_ok = latest.get("Breadth Component Score", 0) > 0
    leadership_ok = latest.get("Leadership Component Score", 0) > 0
    if breadth_ok and leadership_ok:
        return "Breadth and sector leadership both confirm the regime."
    if breadth_ok:
        return "Breadth is supportive, but leadership confirmation is weaker."
    if leadership_ok:
        return "Leadership is supportive, but breadth is narrower."
    return "Breadth and leadership are both missing, so the headline score is less robust."


def _regime_analysis_summary(latest_score: int, latest_regime: str, signals, breadth_read: str) -> str:
    full_support = [signal.name for signal in signals if signal.score == signal.max_score]
    missing = [signal.name for signal in signals if signal.score == 0]
    support_text = ", ".join(full_support) if full_support else "no full-score components"
    missing_text = ", ".join(missing) if missing else "no fully missing components"
    return (
        f"The MR-1 score is {latest_score}/100, placing the dashboard in {latest_regime}. "
        f"Full support comes from {support_text}. Missing pressure comes from {missing_text}. "
        f"{breadth_read}"
    )


def _render_hmm_summary(hmm_result: HMMResult, rule_regime: str) -> None:
    st.subheader("Expanded HMM Market Regime")
    if not hmm_result.available:
        st.warning(hmm_result.warnings[0])
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HMM Regime", hmm_result.regime)
    col2.metric("HMM Confidence", _format_pct(hmm_result.confidence))
    col3.metric("Transition Risk", hmm_result.transition_risk)
    col4.metric("Stress Probability", _format_pct(hmm_result.stress_probability))

    col5, col6, col7 = st.columns(3)
    col5.metric("Bull Probability", _format_pct(hmm_result.bull_probability))
    col6.metric("Neutral Probability", _format_pct(hmm_result.neutral_probability))
    col7.metric("Features Used", f"{hmm_result.feature_count}")

    st.info(_hmm_final_view(rule_regime=rule_regime, hmm_result=hmm_result))
    for warning in hmm_result.warnings:
        st.warning(warning)


def _render_v2_context_cards(
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
    clean_relative_trend: dict | None = None,
) -> None:
    st.subheader("Deeper Market Regime Score Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Regime Persistence**")
        if regime_persistence.available:
            a, b = st.columns(2)
            a.metric("Current Regime", regime_persistence.current_regime)
            b.metric("Days", regime_persistence.days_in_current_regime)
            c, d = st.columns(2)
            c.metric("Maturity", regime_persistence.maturity)
            d.metric("Stability", _format_stability(regime_persistence))
            st.caption(regime_persistence.stability_read)
        else:
            st.info("Regime persistence is unavailable for the selected timeframe.")

    with col2:
        st.markdown("**Options Volatility**")
        col_iv1, col_iv2 = st.columns(2)
        col_iv1.metric("IV Context", options_context.context)
        col_iv2.metric("Current IV", _format_optional_pct(options_context.current_iv))
        col_iv3, col_iv4 = st.columns(2)
        col_iv3.metric("IV Premium", _format_optional_signed_pct(options_context.iv_premium))
        col_iv4.metric("Put/Call Vol", _format_optional_ratio(options_context.put_call_volume_ratio))
        col_fv1, col_fv2 = st.columns(2)
        col_fv1.metric("Finviz Vol W", _format_optional_pct(options_context.finviz_volatility_week))
        col_fv2.metric("Finviz Vol M", _format_optional_pct(options_context.finviz_volatility_month))
        st.caption(options_context.interpretation)
        for warning in options_context.warnings[:2]:
            st.caption(warning)

    with col3:
        st.markdown("**News / Event Catalyst**")
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Catalyst", event_context.catalyst_status)
        col_e2.metric("Event Risk", event_context.event_risk)
        col_e3, col_e4 = st.columns(2)
        col_e3.metric("News 24h", event_context.news_count_24h)
        col_e4.metric("News 7D", event_context.news_count_7d)
        st.caption(event_context.explanation)

    _render_overlay_detail_tables(options_context=options_context, event_context=event_context)


def _ticker_relative_volume_metrics(latest: pd.Series) -> dict:
    return {
        "rvol_20d": _selected_rvol(latest),
        "volume_percentile_1y": _optional_float(latest.get("Volume Percentile")) or _optional_float(latest.get("Volume Percentile 1Y")),
        "volume_z_score": _optional_float(latest.get("Volume Z-Score")),
    }


def _clean_relative_trend_volume_context(latest: pd.Series) -> dict:
    return {
        "context": latest.get("Volume Context"),
        "volume_context": latest.get("Volume Context"),
        "adjustment": latest.get("Volume Adjustment"),
        "volume_adjustment": latest.get("Volume Adjustment"),
        "rvol_medium": latest.get("RVOL Medium"),
        "rvol_20d": latest.get("RVOL 20D"),
        "volume_percentile": latest.get("Volume Percentile"),
        "volume_percentile_1y": latest.get("Volume Percentile 1Y"),
        "daily_return": _safe_latest_return(latest),
    }


def _safe_latest_return(latest: pd.Series) -> float | None:
    return _optional_float(latest.get("Ticker Return")) or _optional_float(latest.get("Daily Return"))


def _relative_trend_asset_type(ticker: str, metadata: dict) -> str:
    symbol = str(ticker or "").upper()
    name = str(metadata.get("company") or metadata.get("name") or "").lower() if metadata else ""
    if symbol.endswith("-USD"):
        return "crypto"
    if any(token in name for token in ["inverse", "bear", "short"]):
        return "inverse ETF"
    if any(token in name for token in ["2x", "3x", "ultra", "leveraged"]):
        return "leveraged ETF"
    return str(metadata.get("quote_type") or metadata.get("source") or "standard")


def _selected_rvol(latest: pd.Series):
    value = _optional_float(latest.get("RVOL Medium"))
    return value if value is not None else _optional_float(latest.get("RVOL 20D"))


def _benchmark_relative_volume_metrics(benchmark_ohlcv: pd.DataFrame) -> dict:
    if benchmark_ohlcv.empty or "Volume" not in benchmark_ohlcv.columns:
        return {}
    volume = pd.to_numeric(benchmark_ohlcv["Volume"], errors="coerce")
    avg_20d = volume.rolling(20).mean()
    rvol_20d = volume / avg_20d
    percentile = volume.rolling(252).apply(lambda values: float((values <= values.iloc[-1]).mean() * 100), raw=False)
    return {
        "rvol_20d": _optional_float(rvol_20d.dropna().iloc[-1]) if not rvol_20d.dropna().empty else None,
        "volume_percentile_1y": _optional_float(percentile.dropna().iloc[-1]) if not percentile.dropna().empty else None,
    }


def _render_relative_context_card(
    relative_context: dict,
    base_score: int,
    adjusted_score: int,
    adjusted_regime: str,
    adjusted_exposure: float,
) -> None:
    st.subheader("Relative Context")
    st.caption("Compares the ticker with the active benchmark and adds a small confirmation/caution read. It does not replace the core MR-1 score.")
    status = relative_context.get("relationship_status", "Unavailable")
    score_adjustment = int(relative_context.get("score_adjustment", 0) or 0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", status)
    col2.metric("Benchmark", relative_context.get("benchmark", "N/A"))
    col3.metric("Relative 20D", _format_optional_signed_pct(relative_context.get("relative_20d")))
    col4.metric("Score Impact", _format_signed_points(score_adjustment))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Relative Extension", _format_zscore(relative_context.get("relative_zscore")))
    col6.metric("Relationship Stability", relative_context.get("correlation_trend", "Unavailable"))
    col7.metric("Beta Trend", relative_context.get("beta_trend", "Unavailable"))
    col8.metric("Volume Confirmation", relative_context.get("volume_confirmation", "Unavailable"))

    adjusted_text = (
        f"Relative-adjusted read: {adjusted_score}/100, {adjusted_regime}, "
        f"{_format_pct(adjusted_exposure)} exposure. Base MR-1 remains {base_score}/100."
    )
    st.info(f"{relative_context.get('interpretation', 'Relative context unavailable.')} {adjusted_text}")
    for warning in relative_context.get("warnings", [])[:2]:
        st.warning(warning)


def _render_clean_relative_trend_card(clean_relative_trend: dict, show_chart: bool = False, compact: bool = False) -> None:
    if not RELATIVE_TREND_QUALITY_CONFIG.get("enabled", True):
        return

    st.subheader("Clean Relative Trend")
    if not clean_relative_trend.get("available", False):
        st.info(clean_relative_trend.get("explanation", "Clean Relative Trend is unavailable."))
        for warning in clean_relative_trend.get("warnings", [])[:3]:
            st.warning(warning)
        return

    score = int(clean_relative_trend.get("score", 0))
    label = str(clean_relative_trend.get("label", "Unavailable"))
    status_class = _trend_quality_status_class(score)
    st.markdown(
        f"""
        <div class="trend-quality-card signal-{status_class}">
            <div class="volume-header">
                <div>
                    <div class="volume-eyebrow">Clean Relative Trend</div>
                    <div class="volume-title">{score} / 100</div>
                    <div class="volume-subtitle">{escape(label)} | {escape(str(clean_relative_trend.get("status", "Unavailable")))}</div>
                </div>
            </div>
            <p>{escape(str(clean_relative_trend.get("explanation", "")))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric(
        "Relative Trend",
        f"{int(clean_relative_trend.get('relative_trend_score', 0))} / {int(clean_relative_trend.get('relative_trend_max', 40))}",
    )
    cols[1].metric(
        "Relationship Stability",
        f"{int(clean_relative_trend.get('relationship_stability_score', 0))} / {int(clean_relative_trend.get('relationship_stability_max', 30))}",
    )
    cols[2].metric(
        "Volume Confirmation",
        f"{int(clean_relative_trend.get('volume_confirmation_score', 0))} / {int(clean_relative_trend.get('volume_confirmation_max', 30))}",
    )
    if not compact:
        drivers = clean_relative_trend.get("positive_drivers", [])
        warnings = clean_relative_trend.get("warnings", [])
        if drivers:
            st.caption("Drivers: " + "; ".join(drivers[:3]))
        for warning in warnings[:3]:
            st.warning(warning)
    if show_chart:
        with st.expander("Relative Strength Quality chart", expanded=False):
            st.plotly_chart(
                _relative_strength_quality_chart(clean_relative_trend),
                width="stretch",
                key=f"clean_relative_trend_chart_{clean_relative_trend.get('timeframe', 'tf')}",
            )


def _render_trend_quality_recommendation(clean_relative_trend: dict, benchmark: str) -> None:
    st.subheader("Trend Quality")
    if not clean_relative_trend.get("available", False):
        st.info(clean_relative_trend.get("explanation", "Clean Relative Trend is unavailable."))
        return

    score = int(clean_relative_trend.get("score", 0))
    label = clean_relative_trend.get("label", "Unavailable")
    st.info(f"Clean Relative Trend Score: {score} / 100 - {label}")
    st.write(clean_relative_trend.get("explanation", "Trend quality is unavailable."))
    guide = pd.DataFrame(
        [
            ("Outperforming?", f"{int(clean_relative_trend.get('relative_trend_score', 0))} / {int(clean_relative_trend.get('relative_trend_max', 40))}", "Is the ticker leading the benchmark?"),
            ("Stable enough?", f"{int(clean_relative_trend.get('relationship_stability_score', 0))} / {int(clean_relative_trend.get('relationship_stability_max', 30))}", f"Is the relationship with {benchmark} reliable enough to trust?"),
            ("Volume confirms?", f"{int(clean_relative_trend.get('volume_confirmation_score', 0))} / {int(clean_relative_trend.get('volume_confirmation_max', 30))}", "Is participation supporting or contradicting the move?"),
        ],
        columns=["Question", "Score", "Read"],
    )
    st.dataframe(guide, use_container_width=True, hide_index=True, height=_table_height(guide))
    drivers = clean_relative_trend.get("positive_drivers", [])
    warnings = clean_relative_trend.get("warnings", [])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Positive Drivers**")
        if drivers:
            for driver in drivers[:5]:
                st.markdown(f"- {driver}")
        else:
            st.markdown("- No clean relative-trend drivers are confirmed.")
    with col2:
        st.markdown("**Warnings / Risks**")
        if warnings:
            for warning in warnings[:5]:
                st.markdown(f"- {warning}")
        else:
            st.markdown("- No major trend-quality warnings are active.")


def _render_overall_recommendation_summary(
    ticker: str,
    benchmark: str,
    latest: pd.Series,
    action: str,
    exposure: float,
    swing_result: SwingResult,
    clean_relative_trend: dict,
) -> None:
    st.subheader("Overall Recommendation Summary")
    clean_score = int(clean_relative_trend.get("score", 0)) if clean_relative_trend.get("available") else None
    clean_label = str(clean_relative_trend.get("label", "Unavailable"))
    relative_read = "Outperforming benchmark" if int(latest.get("Relative Strength Score", 0) or 0) > 0 else "Not outperforming benchmark"
    rows = pd.DataFrame(
        [
            {"Layer": "MR-1 Score / Regime", "Read": f'{int(latest["MR-1 Score"])} / 100 - {latest["Regime"]}', "Takeaway": action},
            {"Layer": "Suggested Exposure", "Read": _format_pct(exposure), "Takeaway": "Position size guide from MR-1"},
            {"Layer": "Swing Score", "Read": f"{swing_result.score} / 100 - {swing_result.setup_label}", "Takeaway": swing_result.action},
            {"Layer": "Volume Context", "Read": str(latest.get("Volume Context", "Unavailable")), "Takeaway": f'{int(latest.get("Volume Adjustment", 0) or 0):+d} MR-1 points'},
            {"Layer": f"Relative Strength vs {benchmark}", "Read": relative_read, "Takeaway": f'{int(latest.get("Relative Strength Score", 0) or 0)} signal points'},
            {"Layer": "Clean Relative Trend", "Read": "Unavailable" if clean_score is None else f"{clean_score} / 100 - {clean_label}", "Takeaway": clean_relative_trend.get("status", "Unavailable")},
        ]
    )
    st.dataframe(rows, use_container_width=True, hide_index=True, height=_table_height(rows))
    st.success(_overall_recommendation_conclusion(ticker, benchmark, latest, swing_result, clean_relative_trend, action))


def _overall_recommendation_conclusion(
    ticker: str,
    benchmark: str,
    latest: pd.Series,
    swing_result: SwingResult,
    clean_relative_trend: dict,
    action: str,
) -> str:
    regime = str(latest.get("Regime", "Unavailable"))
    volume_context = str(latest.get("Volume Context", "Unavailable"))
    relative_positive = int(latest.get("Relative Strength Score", 0) or 0) > 0
    clean_score = int(clean_relative_trend.get("score", 0)) if clean_relative_trend.get("available") else 0

    if regime == "Risk-On" and clean_score >= 80 and swing_result.score >= 70:
        return (
            f"Conclusion: {ticker} has a high-quality trend setup. MR-1 supports risk, "
            f"{ticker} is leading {benchmark}, swing conditions are favorable, and volume context is {volume_context}. "
            "This supports trend-following or swing positioning with normal risk management."
        )
    if regime == "Defensive" or clean_score < 40:
        return (
            f"Conclusion: stay selective. The dashboard action is {action}, and the Clean Relative Trend read is not strong enough "
            "to treat the move as a confirmed setup. Consider reducing exposure or waiting for cleaner relative strength and volume confirmation."
        )
    if relative_positive and clean_score >= 60:
        return (
            f"Conclusion: {ticker} has a usable but imperfect setup. It is outperforming {benchmark}, "
            "but at least one confirmation layer still needs monitoring. Use smaller size or wait for the weaker layer to improve."
        )
    return (
        f"Conclusion: the setup is mixed. MR-1 reads {regime}, but relative strength, swing quality, and volume are not fully aligned. "
        "Use the dashboard as a watchlist signal until confirmation improves."
    )


def _render_relative_context_detail(relative_context: dict) -> None:
    st.subheader("Relative Context")
    if not relative_context.get("available"):
        st.info(relative_context.get("interpretation", "Relative context unavailable."))
        _render_relative_context_explainer(relative_context)
        for warning in relative_context.get("warnings", []):
            st.warning(warning)
        return

    st.info(relative_context.get("interpretation", "Relative context unavailable."))
    _render_relative_context_explainer(relative_context)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", relative_context.get("relationship_status", "Unavailable"))
    col2.metric("Confidence", relative_context.get("confidence", "Low"))
    col3.metric("Score Impact", _format_signed_points(int(relative_context.get("score_adjustment", 0) or 0)))
    col4.metric("Benchmark", relative_context.get("benchmark", "N/A"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("5D Relative", _format_optional_signed_pct(relative_context.get("relative_5d")))
    col6.metric("20D Relative", _format_optional_signed_pct(relative_context.get("relative_20d")))
    col7.metric("60D Relative", _format_optional_signed_pct(relative_context.get("relative_60d")))
    col8.metric("Extension", relative_context.get("relative_extension_label", "Unavailable"))

    history = relative_context.get("history")
    if history is None or history.empty:
        st.info("Not enough data available.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _relative_ratio_chart(history, relative_context),
                width="stretch",
                key=f"relative_ratio_{relative_context.get('ticker')}_{relative_context.get('benchmark')}",
            )
            _render_relative_chart_explanation(
                "Relative Strength Ratio",
                "Shows the ticker divided by the benchmark. Rising means the ticker is leading the benchmark; falling means it is lagging.",
                "Best use: confirm whether price strength is real leadership or just broad-market lift.",
                "Watch out: a stock can rise while this line falls, meaning it is still underperforming the benchmark.",
            )
        with c2:
            st.plotly_chart(
                _relative_zscore_chart(history),
                width="stretch",
                key=f"relative_zscore_{relative_context.get('ticker')}_{relative_context.get('benchmark')}",
            )
            _render_relative_chart_explanation(
                "Relative Extension",
                "Shows how stretched the ticker is versus its own recent relative trend.",
                "Best use: timing. Positive extension supports momentum, but very high readings can mean chase risk.",
                "Watch out: above +2 is extended; below -2 is unusually weak versus recent behavior.",
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                _relative_correlation_chart(history),
                width="stretch",
                key=f"relative_corr_{relative_context.get('ticker')}_{relative_context.get('benchmark')}",
            )
            _render_relative_chart_explanation(
                "Relationship Stability",
                "Shows how closely the ticker and benchmark move together across tactical and broader windows.",
                "Best use: decide whether the selected benchmark is a valid comparison for this ticker.",
                "Watch out: falling or low correlation means relative-strength signals deserve less confidence.",
            )
        with c4:
            st.plotly_chart(
                _relative_beta_chart(history),
                width="stretch",
                key=f"relative_beta_{relative_context.get('ticker')}_{relative_context.get('benchmark')}",
            )
            _render_relative_chart_explanation(
                "Market Sensitivity",
                "Shows how much the ticker tends to move for a benchmark move.",
                "Best use: position sizing and risk. Higher beta means larger swings versus the benchmark.",
                "Watch out: rising beta can improve upside in Risk-On regimes but increases drawdown risk when the market weakens.",
            )

    st.plotly_chart(
        _relative_rvol_chart(relative_context),
        width="stretch",
        key=f"relative_rvol_{relative_context.get('ticker')}_{relative_context.get('benchmark')}",
    )
    st.caption("Relative volume compares current ticker participation with benchmark participation. Strong ticker RVOL can confirm a relative-strength move.")
    _render_relative_chart_explanation(
        "Ticker vs Benchmark Relative Volume",
        "Compares current participation in the ticker against participation in the benchmark.",
        "Best use: confirmation. Strong ticker RVOL supports a relative-strength move when price leadership is already present.",
        "Watch out: high RVOL on down moves can be distribution, not confirmation.",
    )

    warnings = relative_context.get("warnings", [])
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.warning(warning)

    if DEBUG_MODE:
        st.markdown("**Relative Context Debug**")
        st.json(relative_context.get("debug", {}))


def _render_relative_chart_explanation(title: str, meaning: str, best_use: str, watch_out: str) -> None:
    st.markdown(
        f"""
        <div class="chart-note">
            <strong>{escape(title)}</strong><br>
            {escape(meaning)}<br>
            {escape(best_use)}<br>
            {escape(watch_out)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_relative_context_explainer(relative_context: dict) -> None:
    ticker = str(relative_context.get("ticker") or "Ticker").upper()
    benchmark = str(relative_context.get("benchmark") or "Benchmark").upper()
    st.markdown("**How to read this tab**")
    st.caption(
        f"Relative Context asks whether {ticker} is acting better or worse than {benchmark}. "
        "Use it as confirmation and timing context, not as a standalone buy/sell signal."
    )
    guide = pd.DataFrame(
        [
            (
                "Status",
                "Overall relationship read: Supportive, Neutral, Warning, Broken, or Unavailable.",
                "Supportive can reinforce a long setup. Warning/Broken says the benchmark comparison is not confirming cleanly.",
            ),
            (
                "Score Impact",
                "A small modifier applied to the MR-1 read, capped between -10 and +10 points.",
                "Positive impact means relative context helps. Negative impact means caution. The original MR-1 score still remains the base.",
            ),
            (
                "5D / 20D / 60D Relative",
                f"{ticker} performance compared with {benchmark} over short, medium, and longer swing windows.",
                "Positive values mean outperformance. Negative values mean the ticker is lagging the benchmark.",
            ),
            (
                "Relative Extension",
                "How far the ticker/benchmark ratio is from its recent average, shown as a z-score.",
                "Very high readings can mean strength, but also a chase risk. Very low readings show unusual relative weakness.",
            ),
            (
                "Relationship Stability",
                "Ticker/benchmark correlation across tactical windows plus YTD and 52W context.",
                "Stable/high readings make the benchmark useful. Weak/falling readings mean the ticker may be trading on its own drivers.",
            ),
            (
                "Beta Trend",
                "How sensitive the ticker has been to benchmark moves.",
                "Rising/high beta means larger swings versus the benchmark. Falling/low beta means less market sensitivity.",
            ),
            (
                "Volume Confirmation",
                "Whether relative strength or weakness is supported by participation.",
                "Outperformance with stronger ticker volume is more convincing than outperformance on quiet volume.",
            ),
        ],
        columns=["Item", "Meaning", "Trading Read"],
    )
    st.dataframe(guide, use_container_width=True, hide_index=True, height=_table_height(guide))
    st.caption("Rule of thumb: strong relative trend plus stable relationship plus volume confirmation is cleaner than any one metric alone.")


def _render_overlay_detail_tables(
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
) -> None:
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**Options Volatility Detail**")
        options_row = pd.DataFrame(
            [
                {
                    "Ticker": "Selected",
                    "Current IV": options_context.current_iv,
                    "IV Rank": options_context.iv_rank,
                    "IV Percentile": options_context.iv_percentile,
                    "Finviz Vol W": options_context.finviz_volatility_week,
                    "Finviz Vol M": options_context.finviz_volatility_month,
                    "IV Premium": options_context.iv_premium,
                    "Put/Call Vol": options_context.put_call_volume_ratio,
                    "Put/Call OI": options_context.put_call_oi_ratio,
                    "DTE": options_context.days_to_expiration,
                    "Expirations": options_context.expirations_analyzed,
                    "Context": options_context.context,
                }
            ]
        )
        st.dataframe(
            _style_peer_options_table(options_row),
            use_container_width=True,
            hide_index=True,
            height=96,
        )
        if options_context.expiration_table is not None and not options_context.expiration_table.empty:
            st.markdown("**Expiration Dates**")
            st.dataframe(
                _style_options_expiration_table(options_context.expiration_table),
                use_container_width=True,
                hide_index=True,
                height=_table_height(options_context.expiration_table),
            )

    with detail_col2:
        st.markdown("**News / Event Table**")
        if event_context.table.empty:
            st.info("No event data available.")
        else:
            display = event_context.table.head(5).copy()
            display["Published"] = pd.to_datetime(display["Published"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            preferred = ["Published", "Source", "Category", "Sentiment", "Headline", "Link"]
            st.dataframe(
                display[[column for column in preferred if column in display.columns]],
                use_container_width=True,
                hide_index=True,
                height=220,
            )


def _render_combined_risk_overlay(
    latest_regime: str,
    scored: pd.DataFrame,
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
    hmm_result: HMMResult,
) -> None:
    latest = scored.iloc[-1]
    overlay = pd.DataFrame(
        [
            {"Layer": "Market Regime", "Read": latest_regime, "Action": _recommendation_action(int(latest["MR-1 Score"]), latest_regime)},
            {"Layer": "Ticker Regime", "Read": f'{int(latest["MR-1 Score"])} / 100', "Action": _format_pct(float(latest["Exposure"]))},
            {"Layer": "Regime Stability", "Read": _format_stability(regime_persistence), "Action": regime_persistence.maturity},
            {"Layer": "Volume Context", "Read": str(latest.get("Volume Context", "Unavailable")), "Action": f'{int(latest.get("Volume Adjustment", 0)):+d} pts'},
            {"Layer": "IV Context", "Read": options_context.context, "Action": _options_action(options_context)},
            {"Layer": "Event Risk", "Read": event_context.event_risk, "Action": event_context.catalyst_status},
            {"Layer": "HMM Confirmation", "Read": hmm_result.regime if hmm_result.available else "Unavailable", "Action": _hmm_action(hmm_result)},
        ]
    )
    st.dataframe(
        overlay.style.map(_overlay_action_style, subset=["Read", "Action"]),
        use_container_width=True,
        hide_index=True,
        height=_table_height(overlay),
    )
    st.info(_combined_overlay_read(regime_persistence, options_context, event_context))


def _render_recommendation(
    ticker: str,
    benchmark: str,
    scored: pd.DataFrame,
    signals,
    latest_score: int,
    latest_regime: str,
    exposure: float,
    hmm_result: HMMResult,
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
    swing_result: SwingResult,
    clean_relative_trend: dict,
) -> None:
    action = _recommendation_action(latest_score, latest_regime)
    confidence = _confidence_label(latest_score, signals)
    positives = [signal for signal in signals if signal.score > 0]
    negatives = [signal for signal in signals if signal.score == 0]

    st.subheader("Summary Recommendation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendation", action)
    col2.metric("Suggested Exposure", _format_pct(exposure))
    col3.metric("Confidence", confidence)

    st.subheader("Explanation")
    for paragraph in _recommendation_paragraphs(
        ticker=ticker,
        benchmark=benchmark,
        scored=scored,
        latest_regime=latest_regime,
        positives=positives,
        negatives=negatives,
        hmm_result=hmm_result,
    ):
        st.write(paragraph)

    if RELATIVE_TREND_QUALITY_CONFIG.get("show_in_recommendation", True):
        _render_trend_quality_recommendation(clean_relative_trend, benchmark=benchmark)

    _render_overall_recommendation_summary(
        ticker=ticker,
        benchmark=benchmark,
        latest=scored.iloc[-1],
        action=action,
        exposure=exposure,
        swing_result=swing_result,
        clean_relative_trend=clean_relative_trend,
    )

    st.subheader("Volume Explanation")
    st.write(scored.iloc[-1].get("Volume Explanation", "Volume context is unavailable."))

    st.subheader("Combined Risk Overlay")
    _render_combined_risk_overlay(
        latest_regime=latest_regime,
        scored=scored,
        regime_persistence=regime_persistence,
        options_context=options_context,
        event_context=event_context,
        hmm_result=hmm_result,
    )

    st.subheader("HMM Market Filter")
    if hmm_result.available:
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.metric("HMM Regime", hmm_result.regime)
        col_h2.metric("Confidence", _format_pct(hmm_result.confidence))
        col_h3.metric("Transition Risk", hmm_result.transition_risk)
        col_h4.metric("Stress Probability", _format_pct(hmm_result.stress_probability))
        st.info(_hmm_final_view(rule_regime=latest_regime, hmm_result=hmm_result))
    else:
        st.warning(hmm_result.warnings[0])

    st.subheader("Key Drivers")
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Positive**")
        if positives:
            for signal in positives:
                st.markdown(f"- {signal.explanation}")
        else:
            st.markdown("- No positive drivers are active.")
    with col5:
        st.markdown("**Negative**")
        if negatives:
            for signal in negatives:
                st.markdown(f"- {signal.explanation}")
        else:
            st.markdown("- No major negative drivers are active.")

    st.subheader("Risk Warnings")
    for warning in _risk_warnings(
        benchmark=benchmark,
        negatives=negatives,
        regime_persistence=regime_persistence,
        options_context=options_context,
        event_context=event_context,
    ):
        st.warning(warning)


def _render_performance_table(table: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if table.empty:
        st.info("Performance data is unavailable for the selected context.")
        return

    st.dataframe(
        _style_performance_table(table),
        use_container_width=True,
        hide_index=True,
        height=_table_height(table),
    )


def _render_peer_context(
    ticker: str,
    benchmark: str,
    swing_result: SwingResult,
    scored: pd.DataFrame,
    ticker_metadata: dict,
    context,
    timeframe_label: str,
    swing_timeframe: str,
    sensitivity: str,
    lookbacks: dict[str, int],
    volume_timeframe_config: dict,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
) -> None:
    st.subheader("Peers / Sector / Industry / Theme")
    st.caption(f"Peer source: {context.peer_source}")

    latest = scored.iloc[-1]
    volume_table = _build_volume_comparison_table(ticker=ticker, ticker_metadata=ticker_metadata, context=context)
    _render_peer_context_map(ticker=ticker, benchmark=benchmark, context=context)
    _render_scope_badges(
        [
            ("Regime", "Latest reading"),
            ("Volume", "Finviz RVOL + turnover"),
            ("Volatility", f"VIX {lookbacks['vix']}D + ATR/realized vol"),
        ]
    )
    _render_peer_context_analysis(
        ticker=ticker,
        benchmark=benchmark,
        scored=scored,
        swing_result=swing_result,
        volume_table=volume_table,
    )

    if swing_result.performance_table.empty:
        st.info("Performance data is unavailable for this context.")
        return

    rows = swing_result.performance_table
    peer_rows = rows[rows["Type"] == "Peer"]
    context_rows = rows[rows["Type"].isin(["Sector", "Industry", "Theme Proxy", "Sub-Industry Proxy"])]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Market Context Proxies**")
        _render_scope_badges([("Table", "Swing return windows"), ("Selected Swing TF", swing_timeframe)])
        if context_rows.empty:
            st.info("No sector, industry, theme, or sub-industry proxy data is available.")
        else:
            st.dataframe(
                _style_performance_table(context_rows),
                use_container_width=True,
                hide_index=True,
                height=_table_height(context_rows),
            )

    with col2:
        st.markdown("**Peers**")
        _render_scope_badges([("Table", "Swing return windows"), ("Selected Swing TF", swing_timeframe)])
        if peer_rows.empty:
            st.info(f"No peer list is configured for {ticker}; using {benchmark} context only.")
        else:
            st.dataframe(
                _style_performance_table(peer_rows),
                use_container_width=True,
                hide_index=True,
                height=_table_height(peer_rows),
            )

    if not volume_table.empty:
        st.markdown("**Volume / Volatility Comparison**")
        _render_scope_badges(
            [
                ("Volume", "Avg daily + current snapshot"),
                ("Volatility", "Finviz month"),
                ("Sensitivity", sensitivity),
            ]
        )
        st.dataframe(
            _style_volume_comparison_table(volume_table),
            use_container_width=True,
            hide_index=True,
            height=_table_height(volume_table),
        )
        st.plotly_chart(
            _average_float_turnover_chart(volume_table),
            width="stretch",
            key=f"peer_avg_float_turnover_{ticker}_{benchmark}",
        )

    regime_histories = _load_peer_regime_histories(
        symbols=tuple(_relevant_context_symbols(ticker=ticker, benchmark=benchmark, context=context)),
        benchmark=benchmark.upper(),
        period=TIMEFRAMES[timeframe_label],
        lookback_items=tuple(sorted(lookbacks.items())),
        volume_timeframe_items=_hashable_timeframe_config(volume_timeframe_config),
    )
    if regime_histories:
        st.markdown("**MR-1 Regime Score History**")
        chart_assets = _regime_chart_assets(ticker=ticker, benchmark=benchmark, context=context)
        label_by_symbol = {
            symbol: f"{role}: {symbol}" if role != "Peer" else f"Peer: {symbol}"
            for symbol, role, _ in chart_assets
            if symbol in regime_histories
        }
        symbol_by_label = {label: symbol for symbol, label in label_by_symbol.items()}
        default_labels = [
            label
            for symbol, label in label_by_symbol.items()
            if next((group for asset_symbol, _, group in chart_assets if asset_symbol == symbol), "") != "Peer"
        ]
        selected_labels = st.multiselect(
            "Show MR-1 score lines",
            options=list(symbol_by_label.keys()),
            default=default_labels or list(symbol_by_label.keys())[:4],
            key=f"peer_mr1_score_lines_{ticker}_{benchmark}_{timeframe_label}_{sensitivity}",
            help="Toggle ticker, benchmark, sector/industry/theme proxies, and peers on or off.",
        )
        selected_symbols = {symbol_by_label[label] for label in selected_labels if label in symbol_by_label}
        _render_scope_badges([("Dashboard TF", timeframe_label), ("Sensitivity", sensitivity), ("Benchmark", benchmark.upper())])
        if selected_symbols:
            st.plotly_chart(
                _peer_regime_score_chart(
                    regime_histories=regime_histories,
                    ticker=ticker.upper(),
                    timeframe_label=timeframe_label,
                    assets=chart_assets,
                    selected_symbols=selected_symbols,
                ),
                width="stretch",
                key=f"peer_regime_scores_{ticker}_{benchmark}_{timeframe_label}_{sensitivity}",
            )
        else:
            st.info("Select at least one line to show the MR-1 score chart.")
        peer_persistence = build_peer_persistence_table(regime_histories)
        if not peer_persistence.empty:
            st.markdown("**Peer Regime Persistence**")
            _render_scope_badges([("Table", "Current regime duration"), ("Dashboard TF", timeframe_label)])
            st.dataframe(
                _style_peer_persistence_table(peer_persistence),
                use_container_width=True,
                hide_index=True,
                height=_table_height(peer_persistence),
            )

        peer_options = _build_peer_options_table(regime_histories)
        if not peer_options.empty:
            st.markdown("**Peer Options Volatility**")
            _render_scope_badges([("Options", "Multiple expirations"), ("Source", "yfinance chain + Finviz vol")])
            st.dataframe(
                _style_peer_options_table(peer_options),
                use_container_width=True,
                hide_index=True,
                height=_table_height(peer_options),
            )

        peer_events = _build_peer_events_table(regime_histories, peer_options)
        if not peer_events.empty:
            st.markdown("**Peer Event Risk**")
            _render_scope_badges([("News", "24h / 7D"), ("Event Risk", "Earnings + catalyst")])
            st.dataframe(
                _style_peer_events_table(peer_events),
                use_container_width=True,
                hide_index=True,
                height=_table_height(peer_events),
            )


def _render_peer_context_map(ticker: str, benchmark: str, context) -> None:
    rows = pd.DataFrame(
        [
            ("Ticker", ticker.upper(), "Asset being scored."),
            ("Benchmark", benchmark.upper(), "Broad market reference for relative strength."),
            ("Sector ETF", context.sector_etf or "N/A", "Sector-level participation proxy."),
            ("Industry Proxy", context.industry_proxy or "N/A", "Closer industry or subsector proxy."),
            ("Theme Proxy", context.theme_ticker or "N/A", "Thematic risk appetite proxy."),
            ("Peers", ", ".join(context.peers) if context.peers else "N/A", "Comparable tickers used for rank and confirmation."),
        ],
        columns=["Context", "Ticker(s)", "Meaning"],
    )
    st.markdown("**How to read this page**")
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        height=_table_height(rows),
    )


def _render_peer_context_analysis(
    ticker: str,
    benchmark: str,
    scored: pd.DataFrame,
    swing_result: SwingResult,
    volume_table: pd.DataFrame,
) -> None:
    latest = scored.iloc[-1]
    regime = str(latest.get("Regime", "Unavailable"))
    vix_close = _optional_float(latest.get("VIX Close"))
    vix_sma = _optional_float(latest.get("VIX SMA"))
    vix_gap = _relative_gap(vix_close, vix_sma)
    ticker_row = _volume_row(volume_table, ticker.upper())
    ticker_rvol = _optional_float(ticker_row.get("Finviz RVOL")) if ticker_row is not None else None
    peer_rows = volume_table[volume_table["Type"] == "Peer"] if not volume_table.empty else pd.DataFrame()
    peer_median = (
        float(peer_rows["Finviz RVOL"].dropna().median())
        if not peer_rows.empty and not peer_rows["Finviz RVOL"].dropna().empty
        else None
    )
    swing_latest = swing_result.swing_frame.iloc[-1] if not swing_result.swing_frame.empty else pd.Series(dtype=float)
    atr_pct = _optional_float(swing_latest.get("ATR %"))
    realized_vol = _optional_float(swing_latest.get("Realized Volatility")) or _optional_float(swing_latest.get("20D Realized Volatility"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Regime", regime)
    col2.metric("Ticker RVOL", _format_optional_multiple(ticker_rvol))
    col3.metric("Peer Median RVOL", _format_optional_multiple(peer_median))
    col4.metric("VIX vs Average", "N/A" if vix_gap is None else f"{vix_gap:+.1%}")

    st.info(
        _peer_regime_volume_volatility_read(
            ticker=ticker,
            benchmark=benchmark,
            regime=regime,
            latest=latest,
            ticker_rvol=ticker_rvol,
            peer_median=peer_median,
            vix_gap=vix_gap,
            atr_pct=atr_pct,
            realized_vol=realized_vol,
        )
    )


def _peer_regime_volume_volatility_read(
    ticker: str,
    benchmark: str,
    regime: str,
    latest: pd.Series,
    ticker_rvol: float | None,
    peer_median: float | None,
    vix_gap: float | None,
    atr_pct: float | None,
    realized_vol: float | None,
) -> str:
    volume_gap = _relative_gap(ticker_rvol, peer_median)
    volume_read = (
        "cross-sectional volume is unavailable"
        if volume_gap is None
        else "ticker volume is leading peers"
        if volume_gap >= 0.15
        else "ticker volume is lagging peers"
        if volume_gap <= -0.15
        else "ticker volume is in line with peers"
    )
    vix_read = (
        "VIX comparison is unavailable"
        if vix_gap is None
        else "VIX is below its average"
        if vix_gap <= 0
        else "VIX is above its average"
    )
    volatility_read = _ticker_volatility_read(atr_pct=atr_pct, realized_vol=realized_vol)
    trend_read = (
        "above"
        if _optional_float(latest.get("Ticker Trend Score")) and _optional_float(latest.get("Ticker Trend Score")) > 0
        else "below"
    )
    benchmark_read = (
        "supportive"
        if _optional_float(latest.get("Benchmark Trend Score")) and _optional_float(latest.get("Benchmark Trend Score")) > 0
        else "not supportive"
    )

    if regime == "Risk-On":
        regime_read = (
            "Risk-On rewards leadership, so the best setup is ticker volume above peers, sector participation, and contained VIX."
        )
    elif regime == "Neutral":
        regime_read = (
            "Neutral means selectivity matters: peer-leading volume can justify a trade, but weak volume or elevated VIX argues for smaller size."
        )
    else:
        regime_read = (
            "Defensive regimes require proof. Volume spikes need to be separated into accumulation versus distribution, and high volatility should cap position size."
        )

    return (
        f"{ticker.upper()} is {trend_read} its trend filter while {benchmark.upper()} is {benchmark_read}. "
        f"{volume_read}; {vix_read}; {volatility_read}. {regime_read}"
    )


def _ticker_volatility_read(atr_pct: float | None, realized_vol: float | None) -> str:
    atr_text = "ATR is unavailable" if atr_pct is None else f"ATR is {_format_optional_pct(atr_pct)} of price"
    realized_text = (
        "realized volatility is unavailable"
        if realized_vol is None
        else f"selected-window realized volatility is {_format_optional_pct(realized_vol)}"
    )
    if atr_pct is not None and atr_pct >= 0.07:
        risk_text = "volatility risk is high"
    elif atr_pct is not None and atr_pct >= 0.04:
        risk_text = "volatility risk is elevated"
    else:
        risk_text = "volatility risk is manageable"
    return f"{atr_text} and {realized_text}, so {risk_text}"


def _render_swing_trading(
    ticker: str,
    benchmark: str,
    swing_result: SwingResult,
    swing_timeframe: str,
    hmm_result: HMMResult,
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
    clean_relative_trend: dict,
) -> None:
    for warning in swing_result.warnings:
        st.warning(warning)

    st.subheader("Swing Summary")
    st.caption(
        f"Swing Score is timeframe-aware: {swing_result.score_horizon} {swing_result.score_scope} scope. "
        f"Signals: {swing_result.signal_summary}. Shorter swing horizons react faster but are noisier."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", ticker)
    col2.metric("Swing Score", f"{swing_result.score} / 100")
    col3.metric("Setup", swing_result.setup_label)
    col4.metric("Action", swing_result.action)

    focus_return = _selected_ticker_return(swing_result.performance_table, swing_timeframe)
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Suggested Exposure", _format_pct(swing_result.exposure))
    col6.metric("Swing Timeframe", swing_timeframe)
    col7.metric(f"{swing_timeframe} Ticker Return", focus_return)
    col8.warning(f"Main Risk: {swing_result.main_risk}")
    st.info(f"Positive Driver: {swing_result.positive_driver}")
    if RELATIVE_TREND_QUALITY_CONFIG.get("show_in_swing_tab", True):
        _render_clean_relative_trend_card(clean_relative_trend, compact=True)

    st.subheader("HMM Market Risk Filter")
    if hmm_result.available:
        col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
        col_h1.metric("HMM Market Regime", hmm_result.regime)
        col_h2.metric("HMM Confidence", _format_pct(hmm_result.confidence))
        col_h3.metric("Stress Probability", _format_pct(hmm_result.stress_probability))
        col_h4.metric("Transition Risk", hmm_result.transition_risk)
        col_h5.metric("Swing Risk Adjustment", _swing_risk_adjustment(swing_result=swing_result, hmm_result=hmm_result))
    else:
        st.warning(hmm_result.warnings[0])

    st.subheader("Swing Risk Overlays")
    col_o1, col_o2, col_o3, col_o4 = st.columns(4)
    col_o1.metric("Regime Stability", _format_stability(regime_persistence))
    col_o2.metric("Options IV", options_context.context)
    col_o3.metric("Event Risk", event_context.event_risk)
    col_o4.metric("Catalyst", event_context.catalyst_status)
    st.info(_swing_overlay_read(swing_result, regime_persistence, options_context, event_context))
    _render_swing_volatility_card(swing_result.volatility_context)

    st.subheader("Swing Signal Cards")
    if swing_result.signals:
        cols = st.columns(4)
        for index, signal in enumerate(swing_result.signals):
            with cols[index % 4]:
                _render_swing_card(signal)
    else:
        st.info("Swing signals are unavailable for this ticker.")

    _render_performance_table(swing_result.performance_table, title="Swing Performance Table")

    st.subheader("Swing Charts")
    if swing_result.swing_frame.empty:
        st.info("Swing charts are unavailable because indicator data is incomplete.")
    else:
        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(_swing_price_chart(swing_result.swing_frame, ticker), width="stretch", key=f"swing_price_{ticker}_{benchmark}")
        with chart2:
            st.plotly_chart(_swing_relative_strength_chart(swing_result.swing_frame, ticker, benchmark), width="stretch", key=f"swing_rs_{ticker}_{benchmark}")

        chart3, chart4 = st.columns(2)
        with chart3:
            st.plotly_chart(_swing_sector_strength_chart(swing_result.swing_frame, ticker), width="stretch", key=f"swing_sector_{ticker}_{benchmark}")
        with chart4:
            st.plotly_chart(_swing_atr_chart(swing_result.swing_frame, swing_result.volatility_context), width="stretch", key=f"swing_atr_{ticker}_{benchmark}")

    st.subheader("Swing Recommendation Explanation")
    st.write(swing_result.explanation)


def _render_swing_volatility_card(volatility_context: dict) -> None:
    st.markdown("**Swing Volatility**")
    if not volatility_context.get("available"):
        st.info(volatility_context.get("interpretation", "Swing volatility is unavailable."))
        for warning in volatility_context.get("warnings", []):
            st.warning(warning)
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", volatility_context.get("volatility_status", "Unavailable"))
    col2.metric(f"ATR {volatility_context.get('atr_window', 14)}D", _format_optional_number(volatility_context.get("atr")))
    col3.metric("ATR % Price", _format_optional_pct(volatility_context.get("atr_pct_price")))
    col4.metric(
        f"Realized Vol {volatility_context.get('realized_vol_window', 20)}D",
        _format_optional_pct(volatility_context.get("realized_vol")),
    )
    col5.metric("2x ATR Stop Context", _format_optional_pct(volatility_context.get("suggested_stop_distance")))
    st.caption(
        f"Preset: {volatility_context.get('timeframe_preset', DEFAULT_TIMEFRAME_PRESET)}. "
        "Volatility affects sizing, stop distance, and confidence; it is not a bullish/bearish direction signal."
    )
    st.info(volatility_context.get("interpretation", "Swing volatility context is unavailable."))
    for warning in volatility_context.get("warnings", []):
        st.warning(warning)


def _render_swing_card(signal) -> None:
    status_class = signal.status.lower()
    st.markdown(
        f"""
        <div class="signal-card signal-{status_class}">
            <div class="signal-title">{signal.name}</div>
            <div class="signal-status {status_class}">{signal.status}</div>
            <div class="signal-score">{signal.score} / {signal.max_score}</div>
            <div class="signal-row">Current: {signal.current_value}</div>
            <p>{signal.explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _recommendation_action(score: int, regime: str) -> str:
    if regime == "Risk-On":
        return "BUY"
    if regime == "Neutral":
        return "HOLD"
    if score < 25:
        return "AVOID"
    return "REDUCE"


def _confidence_label(score: int, signals) -> str:
    active_signals = sum(1 for signal in signals if signal.score > 0)
    if score >= 85 or score <= 25:
        return "High"
    if active_signals >= 3:
        return "Medium"
    return "Low"


def _recommendation_paragraphs(
    ticker: str,
    benchmark: str,
    scored: pd.DataFrame,
    latest_regime: str,
    positives,
    negatives,
    hmm_result: HMMResult,
) -> list[str]:
    latest = scored.iloc[-1]
    paragraphs = [
        (
            f"{ticker} is currently in a {latest_regime} regime. "
            f"The model combines the ticker trend, {benchmark} trend, VIX, "
            "relative strength, and market breadth/leadership."
        )
    ]

    if latest["Ticker Trend Score"] > 0 and latest["Benchmark Trend Score"] > 0:
        paragraphs.append(
            f"Both {ticker} and {benchmark} are trading above their trend averages, "
            "which supports taking equity risk."
        )
    elif latest["Ticker Trend Score"] > 0:
        paragraphs.append(
            f"{ticker} is above its trend average, but the broader benchmark backdrop "
            "is not fully supportive."
        )
    elif latest["Benchmark Trend Score"] > 0:
        paragraphs.append(
            f"The broader market is supportive through {benchmark}, but {ticker} is "
            "not yet above its own trend average."
        )
    else:
        paragraphs.append(
            f"Neither {ticker} nor {benchmark} is above its trend average, so the setup "
            "leans defensive."
        )

    if latest["VIX Regime Score"] > 0:
        paragraphs.append(
            "Volatility remains contained because VIX is below its average, which helps "
            "support risk-taking."
        )
    else:
        paragraphs.append(
            "Volatility is a warning because VIX is above its average, which can make "
            "breakouts and trend signals less reliable."
        )

    if latest["Relative Strength Score"] > 0:
        paragraphs.append(
            f"{ticker} is outperforming {benchmark} on a relative-strength basis."
        )
    else:
        paragraphs.append(
            f"{ticker} is not outperforming {benchmark} on a relative-strength basis."
        )

    if not positives and negatives:
        paragraphs.append("There are no active positive drivers, so capital preservation comes first.")

    if hmm_result.available:
        paragraphs.append(_hmm_recommendation_paragraph(latest_regime=latest_regime, hmm_result=hmm_result))

    return paragraphs


def _hmm_recommendation_paragraph(latest_regime: str, hmm_result: HMMResult) -> str:
    if hmm_result.regime == "Stress / Risk-Off":
        return (
            "The HMM detects Stress / Risk-Off conditions. Even if the ticker looks strong, "
            "new swing entries should be smaller or avoided."
        )
    if latest_regime == "Risk-On" and hmm_result.regime == "Bull / Calm" and hmm_result.confidence >= 0.75:
        return (
            "The rule-based score is Risk-On, and the HMM confirms a Bull / Calm regime "
            "with high confidence."
        )
    if latest_regime == "Risk-On" and hmm_result.transition_risk in {"Elevated", "High"}:
        return (
            "The rule-based score is Risk-On, but the HMM shows elevated transition risk. "
            "Consider smaller position size or waiting for confirmation."
        )
    return (
        f"The HMM currently shows {hmm_result.regime} with "
        f"{_format_pct(hmm_result.confidence)} confidence and {hmm_result.transition_risk.lower()} transition risk."
    )


def _hmm_final_view(rule_regime: str, hmm_result: HMMResult) -> str:
    if not hmm_result.available:
        return "Final View: HMM unavailable; use the rule score."
    if hmm_result.regime == "Stress / Risk-Off":
        return f"Rule Score: {rule_regime}. HMM Regime: Stress / Risk-Off. Final View: reduce size or avoid new risk."
    if rule_regime == "Risk-On" and hmm_result.regime == "Bull / Calm" and hmm_result.transition_risk == "Low":
        return "Rule Score: Risk-On. HMM Regime: Bull / Calm. Transition Risk: Low. Final View: Risk-On confirmed."
    if rule_regime == "Risk-On" and hmm_result.transition_risk in {"Elevated", "High"}:
        return (
            f"Rule Score: Risk-On. HMM Regime: {hmm_result.regime}. "
            f"Transition Risk: {hmm_result.transition_risk}. Final View: Risk-On, but reduce size."
        )
    return (
        f"Rule Score: {rule_regime}. HMM Regime: {hmm_result.regime}. "
        f"Transition Risk: {hmm_result.transition_risk}."
    )


def _swing_risk_adjustment(swing_result: SwingResult, hmm_result: HMMResult) -> str:
    if not hmm_result.available:
        return "Use MR-1 only"
    if hmm_result.regime == "Stress / Risk-Off":
        return "REDUCE / AVOID"
    if swing_result.setup_label == "Strong Swing Setup" and hmm_result.regime == "Bull / Calm" and hmm_result.confidence >= 0.75:
        return "BUY / ADD"
    if swing_result.setup_label == "Strong Swing Setup" and hmm_result.transition_risk == "Medium":
        return "BUY SMALL"
    if swing_result.setup_label == "Valid Swing Setup" and hmm_result.stress_probability > 0.30:
        return "WATCHLIST"
    if hmm_result.transition_risk in {"Elevated", "High"}:
        return "REDUCE SIZE"
    return swing_result.action


def _risk_warnings(
    benchmark: str,
    negatives,
    regime_persistence: RegimePersistenceResult | None = None,
    options_context: OptionsVolatilityResult | None = None,
    event_context: EventContextResult | None = None,
) -> list[str]:
    warnings = []
    negative_names = {signal.name for signal in negatives}

    if "Volatility" not in negative_names:
        warnings.append("If VIX rises above its average, the regime may shift toward Neutral or Defensive.")
    if "Benchmark Trend" not in negative_names:
        warnings.append(
            f"If {benchmark} falls below its trend average, market support may weaken quickly."
        )
    if "Ticker Trend" not in negative_names:
        warnings.append("If the ticker loses its trend average, reduce position size before the score confirms a deeper regime shift.")
    if "Relative Strength" in negative_names:
        warnings.append("Relative strength is weak, so avoid treating a broad market rally as confirmation for this ticker.")
    if "Market Breadth / Leadership" in negative_names:
        warnings.append("Market breadth or leadership is weak, so the regime may be narrower than the headline score suggests.")
    if regime_persistence and regime_persistence.available and regime_persistence.stability_score is not None and regime_persistence.stability_score < 45:
        warnings.append("Regime persistence is fragile; wait for confirmation or reduce new position size.")
    if options_context and options_context.context in {"IV Event Risk", "IV Stress"}:
        warnings.append("Options IV is warning of event or stress risk, so avoid full-size entries unless that risk is intentional.")
    if event_context and event_context.catalyst_status == "Unexplained volume":
        warnings.append("Volume is elevated without a clear catalyst; treat the flow as less reliable until confirmed.")

    return warnings or ["No major risk warnings are active, but monitor VIX and benchmark trend daily."]


def _format_stability(regime_persistence: RegimePersistenceResult | None) -> str:
    if not regime_persistence or not regime_persistence.available or regime_persistence.stability_score is None:
        return "Unavailable"
    return f"{regime_persistence.stability_score}/100"


def _format_optional_signed_pct(value) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):+.1%}"


def _format_optional_ratio(value) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):.2f}"


def _format_zscore(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):+.2f} sigma"


def _options_action(options_context: OptionsVolatilityResult) -> str:
    if options_context.context in {"IV Event Risk", "IV Stress"}:
        return "Reduce size"
    if options_context.context == "IV Compression / Squeeze Candidate":
        return "Watch expansion"
    if options_context.context == "IV Elevated":
        return "Size selectively"
    if options_context.context in {"IV Calm", "IV Normal"}:
        return "No major IV warning"
    return "Use fallback vol"


def _hmm_action(hmm_result: HMMResult) -> str:
    if not hmm_result.available:
        return "Use rule score"
    if hmm_result.regime == "Stress / Risk-Off":
        return "Reduce risk"
    if hmm_result.transition_risk in {"Elevated", "High"}:
        return "Watch transition"
    return "Confirming"


def _combined_overlay_read(
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
) -> str:
    parts = []
    if regime_persistence.available:
        parts.append(regime_persistence.stability_read)
    if options_context.available:
        parts.append(options_context.interpretation)
    else:
        parts.append("Options data is unavailable, so IV risk is inferred from VIX, ATR, and realized volatility.")
    if event_context.available:
        parts.append(event_context.explanation)
    else:
        parts.append("No catalyst data is available, so news/event confirmation is not part of the final read.")
    return " ".join(parts)


def _swing_overlay_read(
    swing_result: SwingResult,
    regime_persistence: RegimePersistenceResult,
    options_context: OptionsVolatilityResult,
    event_context: EventContextResult,
) -> str:
    adjustments = []
    if options_context.context in {"IV Event Risk", "IV Stress"}:
        adjustments.append("high IV/event risk argues for smaller position size")
    if event_context.catalyst_status == "News-confirmed volume":
        adjustments.append("news-confirmed volume improves confidence")
    if event_context.catalyst_status == "Unexplained volume":
        adjustments.append("unexplained volume requires caution")
    if regime_persistence.available and regime_persistence.stability_score is not None and regime_persistence.stability_score < 45:
        adjustments.append("a fragile or new regime lowers conviction")
    if not adjustments:
        adjustments.append("no major overlay is changing the base swing read")
    return f"Base swing action is {swing_result.action}. Overlay read: {', '.join(adjustments)}."


def _render_charts(
    scored: pd.DataFrame,
    ticker: str,
    benchmark: str,
    options_context: OptionsVolatilityResult | None = None,
    regime_persistence: RegimePersistenceResult | None = None,
) -> None:
    st.subheader("Charts")
    chart1, chart2 = st.columns(2)
    with chart1:
        st.plotly_chart(_price_chart(scored, ticker), width="stretch", key=f"overview_price_{ticker}")
    with chart2:
        st.plotly_chart(_score_chart(scored), width="stretch", key=f"overview_score_{ticker}_{benchmark}")

    chart3, chart4 = st.columns(2)
    with chart3:
        st.plotly_chart(_relative_strength_chart(scored, ticker, benchmark), width="stretch", key=f"overview_rs_{ticker}_{benchmark}")
    with chart4:
        st.plotly_chart(_vix_chart(scored), width="stretch", key=f"overview_vix_{ticker}_{benchmark}")

    chart5, chart6 = st.columns(2)
    with chart5:
        st.plotly_chart(_price_volume_chart(scored, ticker), width="stretch", key=f"overview_price_volume_{ticker}")
    with chart6:
        st.plotly_chart(_relative_volume_chart(scored), width="stretch", key=f"overview_rvol_{ticker}")

    if options_context is not None or regime_persistence is not None:
        chart7, chart8 = st.columns(2)
        with chart7:
            st.plotly_chart(
                _options_iv_chart(options_context),
                width="stretch",
                key=f"overview_options_iv_{ticker}_{benchmark}",
            )
        with chart8:
            st.plotly_chart(
                _regime_time_spent_chart(regime_persistence),
                width="stretch",
                key=f"overview_regime_time_spent_{ticker}_{benchmark}",
            )

        if regime_persistence is not None and regime_persistence.available:
            st.plotly_chart(
                _regime_duration_timeline_chart(regime_persistence),
                width="stretch",
                key=f"overview_regime_duration_{ticker}_{benchmark}",
            )


def _price_chart(scored: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Ticker Close"],
            name=f"{ticker} price",
            line=dict(color="#155e75", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Ticker SMA"],
            name="Trend average",
            line=dict(color="#a16207", width=2),
        )
    )
    return _finish_chart(fig, title=f"{ticker} Price + Trend")


def _price_volume_chart(scored: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if "Volume" in scored.columns:
        fig.add_trace(
            go.Bar(
                x=scored.index,
                y=scored["Volume"],
                name="Daily volume",
                marker_color="rgba(56, 189, 248, 0.35)",
                yaxis="y2",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Ticker Close"],
            name=f"{ticker} price",
            line=dict(color="#38bdf8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Ticker SMA"],
            name="200D trend",
            line=dict(color="#f59e0b", width=2),
        )
    )
    if "Avg Volume 20D" in scored.columns:
        fig.add_trace(
            go.Scatter(
                x=scored.index,
                y=scored["Avg Volume 20D"],
                name="20D avg volume",
                line=dict(color="#94a3b8", width=2),
                yaxis="y2",
            )
        )
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    return _finish_chart(fig, title="Price + Volume")


def _relative_volume_chart(scored: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rvol_column = "RVOL Medium" if "RVOL Medium" in scored.columns else "RVOL 20D"
    medium_window = None
    if "Volume Medium Window" in scored.columns and not scored["Volume Medium Window"].dropna().empty:
        medium_window = int(scored["Volume Medium Window"].dropna().iloc[-1])
    label = f"RVOL {medium_window}D" if medium_window else rvol_column
    if rvol_column in scored.columns:
        fig.add_trace(
            go.Scatter(
                x=scored.index,
                y=scored[rvol_column],
                name=label,
                line=dict(color="#22c55e", width=2),
            )
        )
    fig.add_hline(
        y=1.5,
        line=dict(color="#f59e0b", width=1.4, dash="dash"),
        annotation_text="High volume",
        annotation_font=dict(color="#fde68a", size=11),
    )
    fig.add_hline(
        y=2.5,
        line=dict(color="#f87171", width=1.4, dash="dash"),
        annotation_text="Extreme volume",
        annotation_font=dict(color="#fecaca", size=11),
    )
    return _finish_chart(fig, title="Relative Volume 20D")


def _options_iv_chart(options_context: OptionsVolatilityResult | None) -> go.Figure:
    fig = go.Figure()
    if options_context is None or not options_context.available:
        fig.add_annotation(
            text="Options IV data is unavailable for this ticker.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#cbd5e1", size=13),
        )
        return _finish_chart(fig, title="Options IV vs Realized Volatility")

    labels = ["Current IV", "20D Realized Vol", "IV Premium"]
    values = [
        options_context.current_iv,
        options_context.realized_volatility_20d,
        options_context.iv_premium,
    ]
    colors = ["#38bdf8", "#94a3b8", "#ef4444" if (options_context.iv_premium or 0) > 0 else "#22c55e"]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".1%")
    return _finish_chart(fig, title="Options IV vs Realized Volatility")


def _relative_ratio_chart(history: pd.DataFrame, relative_context: dict) -> go.Figure:
    fig = go.Figure()
    if history.empty or "Relative Ratio" not in history.columns:
        fig.add_annotation(text="Not enough data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Relative Strength Ratio")
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Relative Ratio"],
            name=f"{relative_context.get('ticker')} / {relative_context.get('benchmark')}",
            line=dict(color="#38bdf8", width=2.5),
        )
    )
    return _finish_chart(fig, title="Relative Strength Ratio")


def _relative_strength_quality_chart(clean_relative_trend: dict) -> go.Figure:
    fig = go.Figure()
    history = clean_relative_trend.get("history")
    if history is None or history.empty or "Relative Ratio" not in history.columns:
        fig.add_annotation(text="Not enough data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Relative Strength Quality")

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Relative Ratio"],
            name="Ticker / Benchmark ratio",
            line=dict(color="#38bdf8", width=2.6),
        )
    )
    if "Relative Ratio MA" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Relative Ratio MA"],
                name="Relative ratio average",
                line=dict(color="#f59e0b", width=2.1, dash="dash"),
            )
        )
    if "Rolling Correlation" in history.columns and history["Rolling Correlation"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Rolling Correlation"],
                name="Rolling correlation",
                line=dict(color="#a78bfa", width=1.8, dash="dot"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Correlation", font=dict(color="#c4b5fd")),
                overlaying="y",
                side="right",
                range=[-1, 1],
                showgrid=False,
                tickfont=dict(color="#c4b5fd"),
            )
        )
    details = clean_relative_trend.get("details", {})
    stability = details.get("relationship_stability", {})
    volume = details.get("volume_confirmation", {})
    note = (
        f"Stability: {clean_relative_trend.get('relationship_stability_score', 0)}/"
        f"{clean_relative_trend.get('relationship_stability_max', 30)}"
    )
    corr_short = stability.get("corr_short")
    if corr_short is not None:
        note += f" | Corr: {corr_short:.2f}"
    volume_context = volume.get("context")
    if volume_context:
        note += f" | Volume: {volume_context}"
    fig.add_annotation(
        text=note,
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(color="#cbd5e1", size=12),
        bgcolor="rgba(15, 23, 42, 0.72)",
        bordercolor="rgba(148, 163, 184, 0.26)",
        borderpad=6,
    )
    return _finish_chart(fig, title="Relative Strength Quality")


def _relative_zscore_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if history.empty or "Relative Z-Score" not in history.columns:
        fig.add_annotation(text="Not enough data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Relative Extension")
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Relative Z-Score"],
            name="Relative z-score",
            line=dict(color="#f59e0b", width=2.4),
        )
    )
    fig.add_hline(y=2, line=dict(color="#ef4444", width=1.4, dash="dash"), annotation_text="+2 extended")
    fig.add_hline(y=-2, line=dict(color="#22c55e", width=1.4, dash="dash"), annotation_text="-2 weak")
    return _finish_chart(fig, title="Relative Extension")


def _relative_correlation_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if history.empty:
        fig.add_annotation(text="Not enough data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Rolling Correlation")
    for column, color, dash in [
        ("Correlation 20D", "#38bdf8", "solid"),
        ("Correlation 60D", "#a78bfa", "solid"),
        ("Correlation 120D", "#94a3b8", "solid"),
        ("Correlation YTD", "#22c55e", "dot"),
        ("Correlation 52W", "#f59e0b", "dash"),
    ]:
        if column in history.columns:
            fig.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history[column],
                    name=column,
                    line=dict(color=color, width=2.2, dash=dash),
                )
            )
    fig.add_hline(y=0.25, line=dict(color="#ef4444", width=1.2, dash="dash"), annotation_text="Unstable area")
    return _finish_chart(fig, title="Relationship Stability")


def _relative_beta_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if history.empty:
        fig.add_annotation(text="Not enough data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Rolling Beta")
    for column, color in [("Beta 20D", "#22c55e"), ("Beta 60D", "#f59e0b")]:
        if column in history.columns:
            fig.add_trace(go.Scatter(x=history.index, y=history[column], name=column, line=dict(color=color, width=2.2)))
    fig.add_hline(y=1, line=dict(color="#94a3b8", width=1.2, dash="dash"), annotation_text="Market-like")
    return _finish_chart(fig, title="Market Sensitivity")


def _relative_rvol_chart(relative_context: dict) -> go.Figure:
    fig = go.Figure()
    values = [
        relative_context.get("ticker_rvol_20d"),
        relative_context.get("benchmark_rvol_20d"),
    ]
    labels = [
        f"{relative_context.get('ticker', 'Ticker')} RVOL",
        f"{relative_context.get('benchmark', 'Benchmark')} RVOL",
    ]
    if all(value is None or pd.isna(value) for value in values):
        fig.add_annotation(text="Ticker vs benchmark RVOL is unavailable.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return _finish_chart(fig, title="Ticker vs Benchmark Relative Volume")
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#38bdf8", "#94a3b8"],
            hovertemplate="%{x}: %{y:.2f}x<extra></extra>",
        )
    )
    fig.add_hline(y=1.2, line=dict(color="#f59e0b", width=1.2, dash="dash"), annotation_text="Elevated")
    return _finish_chart(fig, title="Ticker vs Benchmark Relative Volume")


def _regime_time_spent_chart(regime_persistence: RegimePersistenceResult | None) -> go.Figure:
    fig = go.Figure()
    if regime_persistence is None or not regime_persistence.available:
        fig.add_annotation(
            text="Regime persistence is unavailable for this timeframe.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#cbd5e1", size=13),
        )
        return _finish_chart(fig, title="Time Spent by Regime")

    regimes = ["Risk-On", "Neutral", "Defensive"]
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=[regime_persistence.time_spent.get(regime, 0) for regime in regimes],
            marker_color=colors,
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _finish_chart(fig, title="Time Spent by Regime")


def _regime_duration_timeline_chart(regime_persistence: RegimePersistenceResult) -> go.Figure:
    fig = go.Figure()
    timeline = regime_persistence.timeline
    colors = {"Risk-On": "#22c55e", "Neutral": "#f59e0b", "Defensive": "#ef4444"}
    if timeline.empty:
        return _finish_chart(fig, title="Regime Duration Timeline")

    for _, row in timeline.iterrows():
        regime = row["Regime"]
        fig.add_trace(
            go.Scatter(
                x=[row["Start"], row["End"]],
                y=[regime, regime],
                mode="lines",
                line=dict(color=colors.get(regime, "#94a3b8"), width=14),
                name=regime,
                showlegend=False,
                hovertemplate=f"{regime}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            )
        )
    return _finish_chart(fig, title="Regime Duration Timeline")


def _score_chart(scored: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(
        y0=75,
        y1=100,
        fillcolor="#17803d",
        opacity=0.18,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=45,
        y1=75,
        fillcolor="#c99400",
        opacity=0.18,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=0,
        y1=45,
        fillcolor="#b83232",
        opacity=0.18,
        line_width=0,
        layer="below",
    )
    fig.add_hline(
        y=75,
        line=dict(color="#22c55e", width=1.5, dash="dash"),
        annotation_text="Risk-On threshold",
        annotation_position="top left",
        annotation_font=dict(color="#bbf7d0", size=11),
    )
    fig.add_hline(
        y=45,
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        annotation_text="Defensive threshold",
        annotation_position="top left",
        annotation_font=dict(color="#fde68a", size=11),
    )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["MR-1 Score"],
            name="MR-1 Score",
            mode="lines",
            line=dict(color="#38bdf8", width=3),
            hovertemplate="%{x|%Y-%m-%d}<br>MR-1 Score: %{y}<extra></extra>",
        )
    )
    latest_x = scored.index[-1]
    fig.add_annotation(
        x=latest_x,
        y=88,
        text="Risk-On",
        showarrow=False,
        font=dict(color="#bbf7d0", size=12),
        xanchor="right",
    )
    fig.add_annotation(
        x=latest_x,
        y=60,
        text="Neutral",
        showarrow=False,
        font=dict(color="#fde68a", size=12),
        xanchor="right",
    )
    fig.add_annotation(
        x=latest_x,
        y=22,
        text="Defensive",
        showarrow=False,
        font=dict(color="#fecaca", size=12),
        xanchor="right",
    )
    fig.update_yaxes(range=[0, 100])
    return _finish_chart(fig, title="MR-1 Regime Score")


def _relative_strength_chart(scored: pd.DataFrame, ticker: str, benchmark: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Relative Strength"],
            name=f"{ticker}/{benchmark}",
            line=dict(color="#6d28d9", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["Relative Strength SMA"],
            name="Relative strength average",
            line=dict(color="#737373", width=2),
        )
    )
    return _finish_chart(fig, title="Relative Strength")


def _vix_chart(scored: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["VIX Close"],
            name="VIX",
            line=dict(color="#b83232", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scored.index,
            y=scored["VIX SMA"],
            name="VIX average",
            line=dict(color="#737373", width=2),
        )
    )
    return _finish_chart(fig, title="VIX vs Average")


def _render_backtest(scored: pd.DataFrame, ticker: str) -> None:
    st.subheader("Backtest Comparison")
    backtest = run_backtest(scored)
    metrics = backtest_metrics(backtest)
    extended_metrics = _extended_backtest_metrics(backtest, metrics)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Buy & Hold {ticker}", _format_pct(metrics["Buy & Hold Total Return"]))
    col2.metric("MR-1 Model", _format_pct(metrics["MR-1 Total Return"]))
    col3.metric("Return Capture", _format_optional_pct(extended_metrics["Return Capture"]))
    col4.metric("Time in Market", _format_pct(metrics["Time in Market"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Buy & Hold Max Drawdown", _format_pct(metrics["Buy & Hold Max Drawdown"]))
    col6.metric("MR-1 Max Drawdown", _format_pct(metrics["MR-1 Max Drawdown"]))
    col7.metric("Drawdown Reduced By", _format_optional_pct(extended_metrics["Drawdown Reduction"]))
    col8.metric("Volatility Reduced By", _format_optional_pct(extended_metrics["Volatility Reduction"]))

    st.info(
        _backtest_explanation(
            ticker=ticker,
            buy_hold_return=metrics["Buy & Hold Total Return"],
            model_return=metrics["MR-1 Total Return"],
            buy_hold_drawdown=metrics["Buy & Hold Max Drawdown"],
            model_drawdown=metrics["MR-1 Max Drawdown"],
            time_in_market=metrics["Time in Market"],
        )
    )

    st.markdown("**Backtest Decision Read**")
    st.dataframe(
        _style_backtest_decision_table(_backtest_decision_table(ticker=ticker, metrics=metrics, extended_metrics=extended_metrics)),
        use_container_width=True,
        hide_index=True,
        height=250,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Regime Contribution**")
        regime_table = _backtest_regime_table(backtest)
        st.dataframe(
            _style_backtest_regime_table(regime_table),
            use_container_width=True,
            hide_index=True,
            height=_table_height(regime_table),
        )
    with col_b:
        st.markdown("**Consistency Snapshot**")
        consistency_table = _backtest_consistency_table(backtest, metrics, extended_metrics)
        st.dataframe(
            _style_backtest_consistency_table(consistency_table),
            use_container_width=True,
            hide_index=True,
            height=_table_height(consistency_table),
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Buy Hold Equity"],
            name=f"Always invested: {ticker}",
            line=dict(color="#737373", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Always invested: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Model Equity"],
            name="MR-1 exposure rules",
            line=dict(color="#17803d", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>MR-1 model: $%{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Value of starting $1")
    st.plotly_chart(
        _finish_chart(fig, title="Growth of $1: Always Invested vs MR-1 Exposure Rules"),
        width="stretch",
        key=f"backtest_growth_{ticker}",
    )

    chart1, chart2 = st.columns(2)
    with chart1:
        st.plotly_chart(
            _backtest_drawdown_chart(backtest, ticker=ticker),
            width="stretch",
            key=f"backtest_drawdown_{ticker}",
        )
    with chart2:
        st.plotly_chart(
            _backtest_exposure_chart(backtest),
            width="stretch",
            key=f"backtest_exposure_{ticker}",
        )

    st.plotly_chart(
        _backtest_rolling_return_chart(backtest, ticker=ticker),
        width="stretch",
        key=f"backtest_rolling_return_{ticker}",
    )


def _extended_backtest_metrics(backtest: pd.DataFrame, metrics: dict[str, float]) -> dict[str, float | None]:
    buy_hold_return = metrics["Buy & Hold Total Return"]
    model_return = metrics["MR-1 Total Return"]
    buy_hold_drawdown = metrics["Buy & Hold Max Drawdown"]
    model_drawdown = metrics["MR-1 Max Drawdown"]
    buy_hold_vol = metrics["Buy & Hold Volatility"]
    model_vol = metrics["MR-1 Volatility"]
    years = max(len(backtest.index) / 252, 1 / 252)
    monthly = (1 + backtest[["Ticker Return", "Model Return"]]).resample("ME").prod() - 1
    up_days = backtest["Ticker Return"] > 0
    down_days = backtest["Ticker Return"] < 0

    return {
        "Return Capture": _safe_division(model_return, buy_hold_return),
        "Excess Return": model_return - buy_hold_return,
        "Drawdown Reduction": _safe_division(abs(buy_hold_drawdown) - abs(model_drawdown), abs(buy_hold_drawdown)),
        "Volatility Reduction": _safe_division(buy_hold_vol - model_vol, buy_hold_vol),
        "Buy & Hold CAGR": (backtest["Buy Hold Equity"].iloc[-1] / backtest["Buy Hold Equity"].iloc[0]) ** (1 / years) - 1,
        "MR-1 CAGR": (backtest["Model Equity"].iloc[-1] / backtest["Model Equity"].iloc[0]) ** (1 / years) - 1,
        "Buy & Hold Return / Drawdown": _safe_division(buy_hold_return, abs(buy_hold_drawdown)),
        "MR-1 Return / Drawdown": _safe_division(model_return, abs(model_drawdown)),
        "Positive Month Rate": float((monthly["Model Return"] > 0).mean()) if not monthly.empty else None,
        "Best MR-1 Month": float(monthly["Model Return"].max()) if not monthly.empty else None,
        "Worst MR-1 Month": float(monthly["Model Return"].min()) if not monthly.empty else None,
        "Exposure Changes": float((backtest["Model Exposure"].diff().fillna(0) != 0).sum()),
        "Up Capture": _safe_division(backtest.loc[up_days, "Model Return"].mean(), backtest.loc[up_days, "Ticker Return"].mean()),
        "Down Capture": _safe_division(backtest.loc[down_days, "Model Return"].mean(), backtest.loc[down_days, "Ticker Return"].mean()),
    }


def _backtest_decision_table(
    ticker: str,
    metrics: dict[str, float],
    extended_metrics: dict[str, float | None],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Question": "Did MR-1 improve returns?",
                "Reading": _format_optional_signed_pct(extended_metrics["Excess Return"]),
                "Answer": (
                    "MR-1 outperformed buy & hold."
                    if (extended_metrics["Excess Return"] or 0) > 0
                    else f"Buy & hold {ticker} outperformed MR-1."
                ),
            },
            {
                "Question": "Did MR-1 reduce downside?",
                "Reading": _format_optional_pct(extended_metrics["Drawdown Reduction"]),
                "Answer": "Shows how much of the buy-and-hold drawdown MR-1 avoided.",
            },
            {
                "Question": "Was the ride smoother?",
                "Reading": _format_optional_pct(extended_metrics["Volatility Reduction"]),
                "Answer": "Shows whether MR-1 lowered annualized volatility versus staying fully invested.",
            },
            {
                "Question": "How efficient was the risk taken?",
                "Reading": _format_optional_number(extended_metrics["MR-1 Return / Drawdown"]),
                "Answer": "MR-1 return divided by its worst drawdown; higher is better.",
            },
            {
                "Question": "How defensive was the model?",
                "Reading": _format_pct(metrics["Time in Market"]),
                "Answer": "Average exposure from the MR-1 regime rules, not the number of days invested.",
            },
        ]
    )


def _backtest_regime_table(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in backtest.groupby("Regime", sort=False):
        rows.append(
            {
                "Regime": regime,
                "Days": len(group),
                "Avg Exposure": group["Model Exposure"].mean(),
                "Buy & Hold Return": (1 + group["Ticker Return"]).prod() - 1,
                "MR-1 Return": (1 + group["Model Return"]).prod() - 1,
                "Avg Daily MR-1": group["Model Return"].mean(),
                "Worst Day MR-1": group["Model Return"].min(),
            }
        )
    return pd.DataFrame(rows)


def _backtest_consistency_table(
    backtest: pd.DataFrame,
    metrics: dict[str, float],
    extended_metrics: dict[str, float | None],
) -> pd.DataFrame:
    rows = [
        ("Buy & Hold CAGR", extended_metrics["Buy & Hold CAGR"], "Annualized buy-and-hold return."),
        ("MR-1 CAGR", extended_metrics["MR-1 CAGR"], "Annualized MR-1 model return."),
        ("Positive Month Rate", extended_metrics["Positive Month Rate"], "Share of months where MR-1 was positive."),
        ("Best MR-1 Month", extended_metrics["Best MR-1 Month"], "Best calendar month for the MR-1 model."),
        ("Worst MR-1 Month", extended_metrics["Worst MR-1 Month"], "Worst calendar month for the MR-1 model."),
        ("Up Capture", extended_metrics["Up Capture"], "How much MR-1 participated on up days."),
        ("Down Capture", extended_metrics["Down Capture"], "How much MR-1 participated on down days. Lower is better."),
        ("Exposure Changes", extended_metrics["Exposure Changes"], "How often the model changed exposure."),
    ]
    return pd.DataFrame(
        {
            "Metric": [row[0] for row in rows],
            "Value": [row[1] for row in rows],
            "Meaning": [row[2] for row in rows],
        }
    )


def _backtest_drawdown_chart(backtest: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=_drawdown_series(backtest["Buy Hold Equity"]),
            name=f"Always invested: {ticker}",
            line=dict(color="#737373", width=2),
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.16)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=_drawdown_series(backtest["Model Equity"]),
            name="MR-1 exposure rules",
            line=dict(color="#22c55e", width=2),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _finish_chart(fig, title="Drawdown: Peak-to-Trough Pain")


def _backtest_exposure_chart(backtest: pd.DataFrame) -> go.Figure:
    colors = {"Risk-On": "#22c55e", "Neutral": "#f59e0b", "Defensive": "#ef4444"}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Model Exposure"],
            name="MR-1 exposure",
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Exposure: %{y:.0%}<extra></extra>",
        )
    )
    for regime in ["Risk-On", "Neutral", "Defensive"]:
        mask = backtest["Regime"] == regime
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=backtest.index[mask],
                y=backtest.loc[mask, "Model Exposure"],
                name=regime,
                mode="markers",
                marker=dict(color=colors.get(regime, "#94a3b8"), size=5),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}<br>Exposure: %{y:.0%}<extra></extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return _finish_chart(fig, title="MR-1 Exposure Through Time")


def _backtest_rolling_return_chart(backtest: pd.DataFrame, ticker: str, window: int = 63) -> go.Figure:
    fig = go.Figure()
    buy_hold = (1 + backtest["Ticker Return"]).rolling(window).apply(lambda values: values.prod() - 1, raw=True)
    model = (1 + backtest["Model Return"]).rolling(window).apply(lambda values: values.prod() - 1, raw=True)
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=buy_hold,
            name=f"{ticker} rolling 3M",
            line=dict(color="#737373", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>3M return: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=model,
            name="MR-1 rolling 3M",
            line=dict(color="#22c55e", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>3M return: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.update_yaxes(tickformat=".0%")
    return _finish_chart(fig, title="Rolling 3-Month Return")


def _drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def _safe_division(numerator, denominator) -> float | None:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None
    if numerator is None or pd.isna(numerator):
        return None
    return float(numerator / denominator)


def _style_backtest_decision_table(table: pd.DataFrame):
    return (
        table.style.map(_reading_style, subset=["Reading"])
        .set_properties(subset=["Question"], **{"font-weight": "800", "color": "#e5e7eb"})
        .set_properties(subset=["Answer"], **{"color": "#cbd5e1"})
    )


def _style_backtest_regime_table(table: pd.DataFrame):
    pct_columns = ["Avg Exposure", "Buy & Hold Return", "MR-1 Return", "Avg Daily MR-1", "Worst Day MR-1"]
    return (
        table.style.format({column: _format_pct for column in pct_columns}, na_rep="N/A")
        .map(_performance_pct_style, subset=["Buy & Hold Return", "MR-1 Return", "Avg Daily MR-1", "Worst Day MR-1"])
        .set_properties(subset=["Regime"], **{"font-weight": "800", "color": "#e5e7eb"})
    )


def _style_backtest_consistency_table(table: pd.DataFrame):
    display = table.copy()
    percent_metrics = {
        "Buy & Hold CAGR",
        "MR-1 CAGR",
        "Positive Month Rate",
        "Best MR-1 Month",
        "Worst MR-1 Month",
        "Up Capture",
        "Down Capture",
    }
    display["Value"] = display.apply(
        lambda row: _format_optional_pct(row["Value"]) if row["Metric"] in percent_metrics else _format_optional_number(row["Value"]),
        axis=1,
    )
    return (
        display.style.map(_reading_style, subset=["Value"])
        .set_properties(subset=["Metric"], **{"font-weight": "800", "color": "#e5e7eb"})
        .set_properties(subset=["Meaning"], **{"color": "#cbd5e1"})
    )


def _reading_style(value) -> str:
    text = str(value)
    if text.startswith("-"):
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.20); font-weight: 800;"
    if text.startswith("+") or (text.endswith("%") and not text.startswith("0.0")):
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.18); font-weight: 800;"
    return "color: #e5e7eb; background-color: rgba(148, 163, 184, 0.10); font-weight: 700;"


def _backtest_explanation(
    ticker: str,
    buy_hold_return: float,
    model_return: float,
    buy_hold_drawdown: float,
    model_drawdown: float,
    time_in_market: float,
) -> str:
    better_return = "Buy & Hold made more money" if buy_hold_return > model_return else "MR-1 made more money"
    smoother_path = "MR-1 had the smaller drawdown" if model_drawdown > buy_hold_drawdown else "Buy & Hold had the smaller drawdown"
    return (
        f"This is a historical simulation, not a forecast. Gray shows one dollar held in {ticker} the whole time; "
        "green shows one dollar following MR-1 exposure rules. "
        f"In this period, {better_return.lower()}, while {smoother_path.lower()}. "
        f"Time in Market of {_format_pct(time_in_market)} means the MR-1 model was not fully invested all the time."
    )


def _swing_price_chart(frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker Close"], name=f"{ticker} price", line=dict(color="#155e75", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["20D SMA"], name="20D SMA", line=dict(color="#16a34a", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["50D SMA"], name="50D SMA", line=dict(color="#ca8a04", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["200D SMA"], name="200D SMA", line=dict(color="#991b1b", width=2)))
    return _finish_chart(fig, title="Price with 20D / 50D / 200D Averages")


def _swing_relative_strength_chart(frame: pd.DataFrame, ticker: str, benchmark: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker/Benchmark"], name=f"{ticker}/{benchmark}", line=dict(color="#6d28d9", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker/Benchmark 50D SMA"], name="50D RS average", line=dict(color="#737373", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker/Benchmark 63D SMA"], name="63D RS average", line=dict(color="#334155", width=2)))
    return _finish_chart(fig, title="Ticker vs Benchmark Relative Strength")


def _swing_sector_strength_chart(frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker/Sector"], name=f"{ticker}/Sector", line=dict(color="#0f766e", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Ticker/Sector 50D SMA"], name="50D sector RS average", line=dict(color="#737373", width=2)))
    return _finish_chart(fig, title="Ticker vs Sector Relative Strength")


def _swing_atr_chart(frame: pd.DataFrame, volatility_context: dict | None = None) -> go.Figure:
    volatility_context = volatility_context or {}
    atr_window = int(volatility_context.get("atr_window", 14) or 14)
    realized_window = int(volatility_context.get("realized_vol_window", 20) or 20)
    realized_column = "Realized Volatility" if "Realized Volatility" in frame.columns else "20D Realized Volatility"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["ATR %"], name=f"{atr_window}D ATR %", line=dict(color="#b83232", width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame[realized_column], name=f"{realized_window}D realized volatility", line=dict(color="#737373", width=2)))
    fig.update_yaxes(tickformat=".1%")
    return _finish_chart(fig, title="ATR and Realized Volatility")


def _finish_chart(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        height=430,
        margin=dict(l=34, r=28, t=104, b=96),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        ),
        font=dict(color="#e5e7eb"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(domain=[0.0, 0.88])
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _regime_spans(scored: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    start = scored.index[0]
    current_regime = scored["Regime"].iloc[0]

    for idx, regime in scored["Regime"].iloc[1:].items():
        if regime != current_regime:
            spans.append((start, idx, current_regime))
            start = idx
            current_regime = regime

    spans.append((start, scored.index[-1], current_regime))
    return spans


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _format_optional_pct(value) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value):.1%}"


def _format_optional_multiple(value) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value):.1f}x"


def _format_optional_percentile(value) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value):.0f}%"


def _format_optional_number(value) -> str:
    if pd.isna(value):
        return "Unavailable"
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.2f}K"
    return f"{number:,.0f}"


def _format_optional_decimal(value) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.1f}"


def _float_turnover_summary(latest: pd.Series, timeframe_config: dict) -> str:
    parts = []
    for window in timeframe_config.get("float_turnover_windows", []):
        label = "Daily" if int(window) == 1 else f"{int(window)}D"
        column = "Daily Float Turnover" if int(window) == 1 else f"{int(window)}D Float Turnover"
        value = latest.get(column)
        if pd.isna(value):
            continue
        parts.append(f"{label}: {float(value):.1%}")
    return " / ".join(parts) if parts else "Float turnover unavailable"


def _volume_status_class(context: str) -> str:
    if context in {"Accumulation", "Breakout Confirmation"}:
        return "positive"
    if context in {"Distribution", "Panic / Liquidation", "Weak Participation", "Unavailable"}:
        return "warning"
    return "mixed"


def _trend_quality_status_class(score: int) -> str:
    if score >= 80:
        return "positive"
    if score >= 60:
        return "mixed"
    return "warning"


def _filter_scored_by_timeframe(scored: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    if scored.empty:
        return scored

    latest_date = pd.Timestamp(scored.index[-1])
    trading_day_windows = {
        "5D": 5,
        "10D": 10,
        "1M": 21,
        "2M": 42,
        "3M": 63,
        "4M": 84,
        "6M": 126,
        "8M": 168,
        "10M": 210,
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260,
    }

    if timeframe_label in trading_day_windows:
        rows = trading_day_windows[timeframe_label]
        return scored.tail(min(rows, len(scored)))

    if timeframe_label == "QTD":
        start_month = ((latest_date.month - 1) // 3) * 3 + 1
        start_date = pd.Timestamp(latest_date.year, start_month, 1)
        filtered = scored[scored.index >= start_date]
        return filtered if not filtered.empty else scored.tail(1)

    if timeframe_label == "YTD":
        start_date = pd.Timestamp(latest_date.year, 1, 1)
        filtered = scored[scored.index >= start_date]
        return filtered if not filtered.empty else scored.tail(1)

    return scored


RETURN_COLUMNS = ["5D", "10D", "1M", "2M", "3M", "4M", "6M", "8M", "10M", "YTD", "1Y"]


def _format_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    display = _display_performance_table(table)
    for column in RETURN_COLUMNS:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.1%}")
    return display


def _display_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    display = _apply_display_type(table)
    preferred_columns = ["Type", "Ticker", "5D", "10D", "1M", "2M", "3M", "4M", "6M", "8M", "10M", "YTD", "1Y"]
    return display[[column for column in preferred_columns if column in display.columns]]


def _style_performance_table(table: pd.DataFrame):
    display = _display_performance_table(table)
    return_columns = [column for column in RETURN_COLUMNS if column in display.columns]
    formatters = {column: _format_pct for column in return_columns}

    styled = (
        display.style.format(formatters, na_rep="N/A")
        .map(_performance_pct_style, subset=return_columns)
        .set_properties(
            subset=["Ticker", "Type"],
            **{
                "font-weight": "700",
                "color": "#e5e7eb",
            },
        )
    )
    return styled


def _performance_pct_style(value) -> str:
    if pd.isna(value):
        return "color: #94a3b8; background-color: rgba(148, 163, 184, 0.08);"
    if value > 0:
        return "color: #bbf7d0; background-color: rgba(22, 163, 74, 0.22); font-weight: 700;"
    if value < 0:
        return "color: #fecaca; background-color: rgba(220, 38, 38, 0.24); font-weight: 700;"
    return "color: #cbd5e1; background-color: rgba(148, 163, 184, 0.10); font-weight: 600;"


def _table_height(table: pd.DataFrame) -> int:
    return min(520, max(180, 38 * (len(table.index) + 1)))


def _selected_ticker_return(table: pd.DataFrame, swing_timeframe: str) -> str:
    if table.empty or swing_timeframe not in table.columns:
        return "N/A"

    ticker_rows = table[table["Asset"] == "Selected ticker"]
    if ticker_rows.empty:
        return "N/A"

    value = ticker_rows.iloc[0][swing_timeframe]
    return "N/A" if pd.isna(value) else f"{value:.1%}"


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
        }
        .top-bar {
            display: grid;
            grid-template-columns: minmax(130px, 1fr) minmax(140px, 1fr) minmax(180px, 1fr);
            gap: 0.75rem;
            align-items: center;
            border: 1px solid #d7dce0;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            background: #ffffff;
        }
        .app-name {
            font-weight: 800;
            font-size: 1.2rem;
            color: #12313f;
        }
        .top-search,
        .top-date {
            color: #334155;
        }
        .scope-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.25rem 0 0.75rem;
        }
        .scope-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: rgba(15, 23, 42, 0.56);
            color: #cbd5e1;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .scope-badge strong {
            color: #99f6e4;
            font-weight: 850;
        }
        .chart-note {
            border: 1px solid rgba(56, 189, 248, 0.20);
            border-radius: 8px;
            padding: 0.68rem 0.82rem;
            margin: 0.2rem 0 0.9rem;
            background: rgba(15, 23, 42, 0.48);
            color: #cbd5e1;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .chart-note strong {
            color: #bae6fd;
        }
        .trend-quality-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin: 0.25rem 0 0.85rem;
            background: rgba(15, 23, 42, 0.54);
            color: #dbeafe;
        }
        .trend-quality-card.signal-positive {
            border-color: rgba(34, 197, 94, 0.34);
            background: rgba(20, 83, 45, 0.20);
        }
        .trend-quality-card.signal-mixed {
            border-color: rgba(245, 158, 11, 0.34);
            background: rgba(113, 63, 18, 0.18);
        }
        .trend-quality-card.signal-warning {
            border-color: rgba(248, 113, 113, 0.34);
            background: rgba(127, 29, 29, 0.18);
        }
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
            padding: 0.25rem 0 0.55rem !important;
            margin: 0.65rem 0 1.15rem !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            overflow-x: visible !important;
        }
        div[data-testid="stTabs"] div[data-baseweb="tab-border"],
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            display: none !important;
        }
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex: 0 0 auto !important;
            min-height: 3rem !important;
            width: auto !important;
            color: #b6c2d1 !important;
            background:
                linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(8, 13, 24, 0.98)) !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            border-radius: 12px !important;
            padding: 0.66rem 1.08rem !important;
            margin: 0 !important;
            isolation: isolate;
            overflow: hidden;
            box-shadow:
                0 10px 24px rgba(2, 6, 23, 0.30),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            transition: background 170ms ease, border-color 170ms ease, box-shadow 170ms ease, transform 170ms ease, color 170ms ease;
        }
        div[data-testid="stTabs"] button[role="tab"]::before,
        div[data-testid="stTabs"] button[data-baseweb="tab"]::before {
            content: "";
            position: absolute;
            inset: 0;
            z-index: -1;
            background:
                radial-gradient(circle at top left, rgba(103, 232, 249, 0.28), transparent 42%),
                linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(20, 184, 166, 0.05));
            opacity: 0;
            transition: opacity 170ms ease;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            color: #f8fafc !important;
            border-color: rgba(103, 232, 249, 0.48) !important;
            box-shadow: 0 12px 26px rgba(8, 47, 73, 0.30) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button[role="tab"]:hover::before,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover::before {
            opacity: 1;
        }
        div[data-testid="stTabs"] button[role="tab"]:focus-visible,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:focus-visible {
            outline: 2px solid rgba(34, 211, 238, 0.72) !important;
            outline-offset: 2px !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            background:
                linear-gradient(135deg, #0891b2, #0f766e) !important;
            border-color: rgba(165, 243, 252, 0.72) !important;
            box-shadow:
                0 16px 32px rgba(8, 145, 178, 0.34),
                inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after,
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]::after {
            content: "";
            position: absolute;
            left: 18%;
            right: 18%;
            bottom: 0.32rem;
            height: 3px;
            border-radius: 999px;
            background: rgba(240, 253, 250, 0.9);
            box-shadow: 0 0 14px rgba(240, 253, 250, 0.62);
        }
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] button[role="tab"] span,
        div[data-testid="stTabs"] button[data-baseweb="tab"] p,
        div[data-testid="stTabs"] button[data-baseweb="tab"] span {
            color: inherit !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
        div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
        div[data-testid="stAlertContentError"] {
            color: #7f1d1d !important;
            border: 1px solid #fecaca !important;
            border-radius: 8px !important;
            background: #fee2e2 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) *,
        div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) *,
        div[data-testid="stAlertContentError"] * {
            color: #7f1d1d !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
        div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
        div[data-testid="stAlertContentWarning"] {
            color: #7f1d1d !important;
            border: 1px solid #fecaca !important;
            border-radius: 8px !important;
            background: #fee2e2 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) *,
        div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) *,
        div[data-testid="stAlertContentWarning"] * {
            color: #7f1d1d !important;
        }
        [data-testid="stRadio"] {
            margin: 0.65rem 0 1.15rem;
        }
        [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
            align-items: center !important;
            padding: 0.15rem 0 0.35rem !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 3.05rem !important;
            color: #b6c2d1 !important;
            background:
                linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(8, 13, 24, 0.98)) !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            border-radius: 12px !important;
            padding: 0.68rem 1.12rem !important;
            box-shadow:
                0 10px 24px rgba(2, 6, 23, 0.30),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            overflow: hidden !important;
            transition: background 170ms ease, border-color 170ms ease, box-shadow 170ms ease, transform 170ms ease, color 170ms ease;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label::before {
            content: "";
            position: absolute;
            inset: 0;
            z-index: 0;
            background:
                radial-gradient(circle at top left, rgba(103, 232, 249, 0.28), transparent 42%),
                linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(20, 184, 166, 0.05));
            opacity: 0;
            transition: opacity 170ms ease;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label:hover {
            color: #f8fafc !important;
            border-color: rgba(103, 232, 249, 0.58) !important;
            box-shadow: 0 12px 26px rgba(8, 47, 73, 0.30) !important;
            transform: translateY(-1px);
        }
        [data-testid="stRadio"] div[role="radiogroup"] label:hover::before {
            opacity: 1;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            color: #ffffff !important;
            background: linear-gradient(135deg, #0891b2, #0f766e) !important;
            border-color: rgba(165, 243, 252, 0.78) !important;
            box-shadow:
                0 16px 32px rgba(8, 145, 178, 0.34),
                inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
            transform: translateY(-1px);
        }
        [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)::after {
            content: "";
            position: absolute;
            left: 18%;
            right: 18%;
            bottom: 0.32rem;
            height: 3px;
            border-radius: 999px;
            background: rgba(240, 253, 250, 0.92);
            box-shadow: 0 0 14px rgba(240, 253, 250, 0.62);
        }
        [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label input {
            position: absolute !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label p,
        [data-testid="stRadio"] div[role="radiogroup"] label span {
            color: inherit !important;
            font-weight: 850 !important;
            letter-spacing: 0 !important;
            position: relative !important;
            z-index: 1 !important;
        }
        [data-baseweb="select"],
        [data-baseweb="select"] * {
            color: #e5e7eb !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.42);
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.16);
        }
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            color: #f8fafc !important;
            background: rgba(15, 23, 42, 0.92) !important;
            font-weight: 800 !important;
        }
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            border-color: rgba(148, 163, 184, 0.12) !important;
        }
        .regime-card {
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            min-height: 360px;
            background: rgba(15, 23, 42, 0.48);
            color: #e5e7eb;
        }
        .regime-card.active {
            border-color: #22d3ee;
            background: rgba(21, 94, 117, 0.28);
            box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.25);
        }
        .regime-card-title {
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }
        .regime-score {
            color: #a7f3d0;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .regime-section-title {
            color: #bae6fd;
            font-size: 0.85rem;
            font-weight: 800;
            margin-top: 0.8rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }
        .regime-card p,
        .regime-card li {
            color: #d1d5db;
            font-size: 0.9rem;
        }
        .regime-card ul {
            padding-left: 1.1rem;
            margin: 0.25rem 0 0;
        }
        .signal-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 1rem;
            min-height: 260px;
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 24, 0.94)),
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.12), transparent 34%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 12px 28px rgba(2, 6, 23, 0.18);
            color: #e5e7eb;
        }
        .signal-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: #64748b;
        }
        .signal-card.signal-positive {
            border-color: rgba(34, 197, 94, 0.38);
            background:
                linear-gradient(145deg, rgba(10, 30, 24, 0.94), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 36%);
        }
        .signal-card.signal-positive::before {
            background: linear-gradient(90deg, #22c55e, rgba(34, 197, 94, 0.25));
        }
        .signal-card.signal-mixed {
            border-color: rgba(245, 158, 11, 0.38);
            background:
                linear-gradient(145deg, rgba(35, 26, 9, 0.94), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 36%);
        }
        .signal-card.signal-mixed::before {
            background: linear-gradient(90deg, #f59e0b, rgba(245, 158, 11, 0.25));
        }
        .signal-card.signal-warning {
            border-color: rgba(248, 113, 113, 0.4);
            background:
                linear-gradient(145deg, rgba(34, 15, 20, 0.94), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(248, 113, 113, 0.16), transparent 36%);
        }
        .signal-card.signal-warning::before {
            background: linear-gradient(90deg, #f87171, rgba(248, 113, 113, 0.25));
        }
        .signal-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: #f8fafc;
        }
        .signal-status {
            display: inline-block;
            font-weight: 700;
            margin-bottom: 0.75rem;
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            background: rgba(15, 23, 42, 0.72);
        }
        .signal-status.positive {
            color: #86efac;
        }
        .signal-status.warning {
            color: #fca5a5;
        }
        .signal-status.mixed {
            color: #fcd34d;
        }
        .signal-score {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: #ffffff;
        }
        .signal-row {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .signal-card p {
            color: #aebccb;
            font-size: 0.9rem;
            margin-top: 0.75rem;
        }
        .volume-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.26);
            border-radius: 8px;
            padding: 1.1rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.14), transparent 34%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 14px 30px rgba(2, 6, 23, 0.20);
            color: #e5e7eb;
        }
        .volume-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, #14b8a6, rgba(20, 184, 166, 0.18));
        }
        .volume-card.signal-positive {
            border-color: rgba(34, 197, 94, 0.34);
        }
        .volume-card.signal-mixed {
            border-color: rgba(245, 158, 11, 0.34);
        }
        .volume-card.signal-warning {
            border-color: rgba(248, 113, 113, 0.36);
        }
        .volume-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        .volume-eyebrow {
            color: #99f6e4;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .volume-title {
            color: #f8fafc;
            font-size: 1.35rem;
            line-height: 1.2;
            font-weight: 850;
            margin-top: 0.15rem;
        }
        .volume-subtitle {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        .volume-adjustment {
            min-width: 4.2rem;
            text-align: center;
            border-radius: 8px;
            padding: 0.45rem 0.7rem;
            font-size: 1.15rem;
            font-weight: 850;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.26);
        }
        .volume-adjustment.positive {
            color: #86efac;
            border-color: rgba(34, 197, 94, 0.34);
        }
        .volume-adjustment.mixed {
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.34);
        }
        .volume-adjustment.warning {
            color: #fca5a5;
            border-color: rgba(248, 113, 113, 0.36);
        }
        .volume-metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 1rem;
        }
        .volume-metric {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 8px;
            padding: 0.75rem;
            background: rgba(15, 23, 42, 0.58);
            min-height: 6.1rem;
        }
        .volume-metric span,
        .volume-metric small {
            display: block;
            color: #94a3b8;
            font-size: 0.78rem;
        }
        .volume-metric strong {
            display: block;
            color: #f8fafc;
            font-size: 1.18rem;
            line-height: 1.2;
            margin: 0.35rem 0 0.2rem;
        }
        .volume-analysis {
            border-left: 3px solid rgba(20, 184, 166, 0.65);
            padding-left: 0.85rem;
            margin-bottom: 0.9rem;
        }
        .volume-analysis p {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.48;
            margin: 0 0 0.55rem;
        }
        .volume-foot {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .volume-foot span,
        .volume-note {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.58);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .volume-note {
            display: inline-block;
            margin-top: 0.55rem;
            border-radius: 8px;
            color: #fcd34d;
        }
        .context-map {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 34%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 12px 28px rgba(2, 6, 23, 0.18);
        }
        .context-map-title {
            color: #f8fafc;
            font-weight: 850;
            font-size: 1rem;
            margin-bottom: 0.8rem;
        }
        .context-map-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
        }
        .context-map-item {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.7rem;
            min-height: 6.5rem;
            background: rgba(15, 23, 42, 0.55);
        }
        .context-map-item span,
        .context-map-item small {
            display: block;
            color: #94a3b8;
            font-size: 0.78rem;
        }
        .context-map-item strong {
            display: block;
            color: #f8fafc;
            font-size: 0.95rem;
            line-height: 1.25;
            margin: 0.25rem 0;
            overflow-wrap: anywhere;
        }
        @media (max-width: 800px) {
            .top-bar {
                grid-template-columns: 1fr;
            }
            .volume-header {
                flex-direction: column;
            }
            .volume-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .context-map-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 520px) {
            .volume-metric-grid {
                grid-template-columns: 1fr;
            }
            .context-map-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
