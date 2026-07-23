from __future__ import annotations

import pandas as pd

try:
    from config import MANUAL_FLOAT_SHARES
except Exception:
    MANUAL_FLOAT_SHARES = {}


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
        return _share_count_result(True, finviz_float, "shares_float", "finviz", True)

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
        return _share_count_result(True, yfinance_float, "floatShares", "yfinance", True)

    manual_float = _number_or_none(manual_float_map.get(ticker))
    if manual_float:
        return _share_count_result(True, manual_float, "shares_float", "manual_config", True)

    shares_outstanding = _number_or_none(
        yfinance_metadata.get("yfinance_shares_outstanding")
        or yfinance_metadata.get("sharesOutstanding")
        or yfinance_metadata.get("shares_outstanding")
        or finviz_metadata.get("shares_outstanding")
    )
    if shares_outstanding:
        return _share_count_result(
            True,
            shares_outstanding,
            "sharesOutstanding",
            "yfinance",
            False,
            "Using shares outstanding proxy because shares float is unavailable.",
        )

    market_cap = _number_or_none(
        yfinance_metadata.get("yfinance_market_cap")
        or yfinance_metadata.get("marketCap")
        or yfinance_metadata.get("market_cap")
        or finviz_metadata.get("market_cap")
    )
    price = _number_or_none(
        latest_price
        or yfinance_metadata.get("yfinance_price")
        or yfinance_metadata.get("price")
        or finviz_metadata.get("price")
    )
    if market_cap and price:
        estimated = market_cap / price
        if estimated:
            return _share_count_result(
                True,
                estimated,
                "estimated_shares_outstanding",
                "estimated",
                False,
                "Share turnover proxy estimated from market cap / price. Use as directional only.",
            )

    return _share_count_result(False, None, None, None, False, "No share-count denominator available.")


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
    avg_daily_turnover = _latest_number(
        volume.rolling(volume_window, min_periods=max(1, min(volume_window, len(volume)))).mean() / denominator
    )
    five_day_turnover = _latest_number(
        volume.rolling(five_day_window, min_periods=max(1, min(five_day_window, len(volume)))).sum() / denominator
    )
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
