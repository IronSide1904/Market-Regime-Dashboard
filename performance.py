from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


PERFORMANCE_WINDOWS = {
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
}


@dataclass(frozen=True)
class AssetPerformance:
    label: str
    ticker: str
    asset_type: str
    returns: dict[str, float | None]


def download_close(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    symbols = sorted({ticker for ticker in tickers if ticker})
    if not symbols:
        return pd.DataFrame()

    downloaded = yf.download(
        symbols,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    return _extract_field(downloaded, symbols=symbols, field="Close")


def calculate_performance_rows(
    close: pd.DataFrame,
    assets: list[tuple[str, str, str]],
) -> pd.DataFrame:
    rows = []
    for label, ticker, asset_type in assets:
        if ticker not in close.columns or close[ticker].dropna().empty:
            continue

        series = close[ticker].dropna()
        returns = calculate_returns(series)
        row = {"Asset": label, "Ticker": ticker, "Type": asset_type}
        row.update(returns)
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_returns(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {label: None for label in [*PERFORMANCE_WINDOWS, "QTD", "YTD"]}

    returns = {label: _lookback_return(clean, days) for label, days in PERFORMANCE_WINDOWS.items()}
    returns["QTD"] = _period_to_date_return(clean, period="quarter")
    returns["YTD"] = _period_to_date_return(clean, period="year")
    return returns


def _lookback_return(series: pd.Series, days: int) -> float | None:
    if len(series) <= days:
        return None
    return float(series.iloc[-1] / series.iloc[-days - 1] - 1)


def _period_to_date_return(series: pd.Series, period: str) -> float | None:
    latest_date = pd.Timestamp(series.index[-1])
    if period == "quarter":
        start_month = ((latest_date.month - 1) // 3) * 3 + 1
        start_date = pd.Timestamp(year=latest_date.year, month=start_month, day=1)
    else:
        start_date = pd.Timestamp(year=latest_date.year, month=1, day=1)

    period_series = series[series.index >= start_date]
    if len(period_series) < 2:
        return None
    return float(series.iloc[-1] / period_series.iloc[0] - 1)


def _extract_field(downloaded: pd.DataFrame, symbols: list[str], field: str) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if field not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        data = downloaded[field].copy()
    else:
        if field not in downloaded.columns:
            return pd.DataFrame()
        data = downloaded[[field]].copy()
        data.columns = [symbols[0]]

    return data.reindex(columns=symbols)
