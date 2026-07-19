from __future__ import annotations

import pandas as pd


def resolve_comparison_ticker(
    selected_ticker: str,
    benchmark_ticker: str,
    peer_override_mode: str = "Auto",
    peer_override_ticker: str | None = None,
) -> dict:
    selected = _normalize_ticker(selected_ticker)
    benchmark = _normalize_ticker(benchmark_ticker)
    requested = _normalize_ticker(peer_override_ticker or "")
    mode = _normalize_mode(peer_override_mode)

    if mode in {"Suggested Peer", "Custom Ticker", "Custom"}:
        if not requested:
            return _fallback(benchmark, "Benchmark", "Peer override ticker is blank. Using benchmark instead.")
        if requested == selected:
            return _fallback(
                benchmark,
                "Benchmark",
                "Peer override equals selected ticker. Using benchmark instead.",
            )
        return {
            "comparison_ticker": requested,
            "comparison_type": "Peer Override",
            "is_override_active": True,
            "warning": None,
            "requested_ticker": requested,
            "mode": mode,
        }

    return {
        "comparison_ticker": benchmark,
        "comparison_type": "Benchmark",
        "is_override_active": False,
        "warning": None,
        "requested_ticker": requested or None,
        "mode": mode,
    }


def validate_comparison_ohlcv(
    selected_ticker: str,
    comparison_ticker: str,
    comparison_ohlcv: pd.DataFrame,
    min_required_rows: int = 60,
) -> str | None:
    selected = _normalize_ticker(selected_ticker)
    comparison = _normalize_ticker(comparison_ticker)
    if not comparison:
        return "Comparison ticker is blank."
    if comparison == selected:
        return "Comparison ticker equals selected ticker."
    if comparison_ohlcv is None or comparison_ohlcv.empty:
        return f"{comparison} data could not be loaded."
    if "Close" not in comparison_ohlcv.columns:
        return f"{comparison} data does not include close prices."
    close = pd.to_numeric(comparison_ohlcv["Close"], errors="coerce").dropna()
    if close.empty:
        return f"{comparison} close prices are unavailable."
    if len(close.index) < int(min_required_rows):
        return f"{comparison} has only {len(close.index)} valid rows; at least {min_required_rows} are required."
    return None


def _fallback(benchmark: str, comparison_type: str, warning: str) -> dict:
    return {
        "comparison_ticker": benchmark,
        "comparison_type": comparison_type,
        "is_override_active": False,
        "warning": warning,
        "requested_ticker": None,
        "mode": "Auto",
    }


def _normalize_mode(mode: str | None) -> str:
    value = str(mode or "Auto").strip()
    if value == "Disabled":
        return "Disabled"
    if value in {"Suggested Peer", "Custom Ticker", "Custom"}:
        return value
    return "Auto"


def _normalize_ticker(ticker: str | None) -> str:
    return str(ticker or "").strip().upper().replace("/", "-").replace("\\", "-")
