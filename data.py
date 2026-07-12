from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import pandas as pd
import yfinance as yf

from config import MANUAL_FLOAT_SHARES, MARKET_TICKERS, MIN_TRADING_DAYS, RELATIVE_CONTEXT_CONFIG, SECTOR_ETF_MAP
from finviz_fetcher import fetch_finviz_ticker_snapshot, remove_dead_local_proxy

YFINANCE_CACHE_DIR = Path(tempfile.gettempdir()) / "mr1-yfinance-cache" / f"{os.getpid()}-{uuid4().hex}"
_YFINANCE_CACHE_CONFIGURED = False


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame
    ticker_ohlcv: pd.DataFrame
    warnings: list[str]


def _extract_field(downloaded: pd.DataFrame, symbols: dict[str, str], field: str) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if field in downloaded.columns.get_level_values(0):
            values = downloaded[field].copy()
        elif field == "Close" and "Adj Close" in downloaded.columns.get_level_values(0):
            values = downloaded["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        column = field if field in downloaded.columns else "Adj Close" if field == "Close" and "Adj Close" in downloaded.columns else None
        if column is None:
            return pd.DataFrame()
        values = downloaded[[column]].copy()
        values.columns = [next(iter(set(symbols.values())))]

    logical_values = pd.DataFrame(index=values.index)
    for logical_name, yf_symbol in symbols.items():
        if yf_symbol in values.columns:
            logical_values[logical_name] = values[yf_symbol]
        else:
            logical_values[logical_name] = pd.NA

    return logical_values


def _extract_close(downloaded: pd.DataFrame, symbols: dict[str, str]) -> pd.DataFrame:
    return _extract_field(downloaded, symbols, "Close")


def _extract_ticker_ohlcv(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame:
    fields = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        extracted = _extract_field(downloaded, {"Ticker": ticker}, field)
        if "Ticker" in extracted.columns:
            fields[field] = extracted["Ticker"]

    if not fields:
        return pd.DataFrame()
    return pd.DataFrame(fields).dropna(how="all")


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace("/", "-").replace("\\", "-")


def fetch_market_data(ticker: str, benchmark: str, period: str) -> MarketData:
    remove_dead_local_proxy()
    _configure_yfinance_cache()
    ticker = normalize_ticker(ticker)
    benchmark = normalize_ticker(benchmark)
    symbols = _symbols_for(ticker=ticker, benchmark=benchmark)
    downloaded = yf.download(
        sorted(set(symbols.values())),
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )

    close = _extract_close(downloaded, symbols)
    ticker_ohlcv = _extract_ticker_ohlcv(downloaded, ticker)
    warnings = validate_market_data(close, required_columns=list(symbols.keys()))
    warnings.extend(validate_volume_data(ticker_ohlcv))
    return MarketData(close=close, ticker_ohlcv=ticker_ohlcv, warnings=warnings)


def get_comparison_benchmark(
    ticker: str,
    metadata: dict,
    user_benchmark: str | None = None,
) -> str:
    normalized_user = normalize_ticker(user_benchmark or "")
    normalized_ticker = normalize_ticker(ticker)
    if normalized_user and normalized_user != normalized_ticker:
        return normalized_user

    sector = metadata.get("sector") if metadata else None
    sector_benchmark = SECTOR_ETF_MAP.get(str(sector)) if sector else None
    if sector_benchmark:
        return sector_benchmark

    return RELATIVE_CONTEXT_CONFIG["default_benchmark"]


def get_benchmark_ohlcv(benchmark: str, period: str = "1y") -> pd.DataFrame:
    remove_dead_local_proxy()
    _configure_yfinance_cache()
    symbol = normalize_ticker(benchmark or RELATIVE_CONTEXT_CONFIG["default_benchmark"])
    try:
        downloaded = yf.download(
            symbol,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()

    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        fields = {}
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if field in downloaded.columns.get_level_values(0):
                values = downloaded[field]
                if isinstance(values, pd.DataFrame):
                    fields[field] = values[symbol] if symbol in values.columns else values.iloc[:, 0]
                else:
                    fields[field] = values
        return pd.DataFrame(fields).dropna(how="all") if fields else pd.DataFrame()

    fields = {field: downloaded[field] for field in ["Open", "High", "Low", "Close", "Volume"] if field in downloaded.columns}
    return pd.DataFrame(fields).dropna(how="all") if fields else pd.DataFrame()


def get_ticker_metadata(ticker: str) -> dict:
    normalized_ticker = normalize_ticker(ticker)
    finviz_snapshot = fetch_finviz_ticker_snapshot(normalized_ticker)
    if finviz_snapshot.get("available"):
        fallback = _yfinance_metadata(normalized_ticker)
        for key, value in fallback.items():
            if key in {"available", "source", "ticker", "error"}:
                continue
            if _is_missing_metadata_value(finviz_snapshot.get(key)) and not _is_missing_metadata_value(value):
                finviz_snapshot[key] = value
        if fallback.get("source") == "yfinance":
            finviz_snapshot["fallback_source"] = "yfinance"
        return finviz_snapshot

    fallback = _yfinance_metadata(normalized_ticker)
    fallback["finviz_available"] = False
    fallback["finviz_error"] = finviz_snapshot.get("error")
    return fallback


def _is_missing_metadata_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _yfinance_metadata(ticker: str) -> dict:
    remove_dead_local_proxy()
    _configure_yfinance_cache()
    result = {
        "available": False,
        "source": "unavailable",
        "ticker": ticker,
        "company": None,
        "sector": None,
        "industry": None,
        "country": None,
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "peg": None,
        "ps": None,
        "pb": None,
        "pc": None,
        "pfcf": None,
        "shares_outstanding": None,
        "shares_float": MANUAL_FLOAT_SHARES.get(ticker),
        "float_percent": None,
        "short_float": None,
        "short_ratio": None,
        "short_interest": None,
        "roa": None,
        "roe": None,
        "roi": None,
        "current_ratio": None,
        "quick_ratio": None,
        "lt_debt_to_equity": None,
        "debt_to_equity": None,
        "gross_margin": None,
        "operating_margin": None,
        "profit_margin": None,
        "relative_volume": None,
        "average_volume": None,
        "volume": None,
        "price": None,
        "change": None,
        "atr": None,
        "beta": None,
        "volatility_week": None,
        "volatility_month": None,
        "sma20": None,
        "sma50": None,
        "sma200": None,
        "high52w": None,
        "low52w": None,
        "rsi": None,
        "change_from_open": None,
        "gap": None,
        "earnings_date": None,
        "trades": None,
        "after_hours_volume": None,
        "error": None,
    }

    try:
        info = yf.Ticker(ticker).get_info()
    except Exception as exc:
        result["error"] = str(exc) or exc.__class__.__name__
        return result

    result.update(
        {
            "available": True,
            "source": "yfinance",
            "company": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg": info.get("pegRatio"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "pb": info.get("priceToBook"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "shares_float": info.get("floatShares") or result["shares_float"],
            "short_float": _fraction_or_none(info.get("shortPercentOfFloat")),
            "short_ratio": info.get("shortRatio"),
            "short_interest": info.get("sharesShort"),
            "roa": _fraction_or_none(info.get("returnOnAssets")),
            "roe": _fraction_or_none(info.get("returnOnEquity")),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "gross_margin": _fraction_or_none(info.get("grossMargins")),
            "operating_margin": _fraction_or_none(info.get("operatingMargins")),
            "profit_margin": _fraction_or_none(info.get("profitMargins")),
            "average_volume": info.get("averageVolume"),
            "volume": info.get("volume"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change": info.get("regularMarketChangePercent"),
            "beta": info.get("beta"),
        }
    )
    return result


def _fraction_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _configure_yfinance_cache() -> None:
    global _YFINANCE_CACHE_CONFIGURED
    if _YFINANCE_CACHE_CONFIGURED:
        return
    try:
        YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
            yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR))
        else:
            yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
        _YFINANCE_CACHE_CONFIGURED = True
    except Exception:
        pass


def validate_market_data(close: pd.DataFrame, required_columns: list[str]) -> list[str]:
    warnings: list[str] = []

    if close.empty:
        return ["Ticker not found or data unavailable."]

    for ticker in required_columns:
        if ticker not in close.columns:
            warnings.append(f"{ticker} is missing from the downloaded data.")
            continue

        valid_count = int(close[ticker].dropna().shape[0])
        if valid_count < MIN_TRADING_DAYS:
            warnings.append(
                f"{ticker} has only {valid_count} valid trading days; "
                f"at least {MIN_TRADING_DAYS} are required."
            )

    return warnings


def validate_volume_data(ticker_ohlcv: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if ticker_ohlcv.empty or "Volume" not in ticker_ohlcv.columns:
        return ["Volume data is unavailable; volume context will be skipped."]

    volume = pd.to_numeric(ticker_ohlcv["Volume"], errors="coerce").dropna()
    if volume.empty:
        warnings.append("Volume data is not numeric; volume context will be skipped.")
    elif (volume == 0).all():
        warnings.append("Volume data is entirely zero; volume context will be skipped.")
    elif volume.shape[0] < MIN_TRADING_DAYS:
        warnings.append("Volume history has fewer than 252 trading days; volume context will be skipped.")
    elif pd.isna(volume.iloc[-1]):
        warnings.append("Current volume is unavailable; volume context will be skipped.")

    return warnings


def _symbols_for(ticker: str, benchmark: str) -> dict[str, str]:
    symbols = {
        "Ticker": ticker,
        "Benchmark": benchmark,
        "VIX": MARKET_TICKERS["VIX"],
        "RSP": MARKET_TICKERS["RSP"],
        "SPY": MARKET_TICKERS["SPY"],
        "XLK": MARKET_TICKERS["XLK"],
        "XLU": MARKET_TICKERS["XLU"],
    }

    if benchmark == "QQQ":
        symbols["QQQ"] = MARKET_TICKERS["QQQ"]

    return symbols
