from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DEBUG_MODE, FINVIZ_COLUMNS, FINVIZ_CONFIG, FINVIZ_DISCOVERY_COLUMNS


load_dotenv()
load_dotenv(Path(__file__).with_name("Finviz API Token.env"), override=False)

NORMALIZED_FIELDS = [
    "ticker",
    "company",
    "sector",
    "industry",
    "country",
    "market_cap",
    "pe",
    "forward_pe",
    "peg",
    "ps",
    "pb",
    "pc",
    "pfcf",
    "shares_outstanding",
    "shares_float",
    "float_percent",
    "short_float",
    "short_ratio",
    "short_interest",
    "roa",
    "roe",
    "roi",
    "current_ratio",
    "quick_ratio",
    "lt_debt_to_equity",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
    "change",
    "atr",
    "beta",
    "volatility_week",
    "volatility_month",
    "sma20",
    "sma50",
    "sma200",
    "high52w",
    "low52w",
    "rsi",
    "change_from_open",
    "gap",
    "earnings_date",
    "trades",
    "after_hours_volume",
]

COLUMN_ALIASES = {
    "ticker": ["ticker", "symbol"],
    "company": ["company", "company_name", "name"],
    "sector": ["sector"],
    "industry": ["industry"],
    "country": ["country"],
    "market_cap": ["market_cap", "marketcap", "market_capitalization"],
    "pe": ["p_e", "pe"],
    "forward_pe": ["forward_p_e", "forward_pe"],
    "peg": ["peg"],
    "ps": ["p_s", "ps"],
    "pb": ["p_b", "pb"],
    "pc": ["p_cash", "p_c", "pc"],
    "pfcf": ["p_free_cash_flow", "p_fcf", "pfcf"],
    "shares_outstanding": ["shares_outstanding", "shs_outstand", "shs_outstanding", "shares_out"],
    "shares_float": ["shares_float", "shs_float", "float_shares", "float"],
    "float_percent": ["float_outstanding", "float_percent", "float_pct", "float"],
    "short_float": ["short_float", "float_short", "short_interest_share", "short_float_percent", "short_float_pct"],
    "short_ratio": ["short_ratio", "short_interest_ratio"],
    "short_interest": ["short_interest", "shares_short"],
    "roa": ["return_on_assets", "roa"],
    "roe": ["return_on_equity", "roe"],
    "roi": ["return_on_invested_capital", "roi", "roic"],
    "current_ratio": ["current_ratio"],
    "quick_ratio": ["quick_ratio"],
    "lt_debt_to_equity": ["lt_debt_equity", "lt_debt_to_equity"],
    "debt_to_equity": ["debt_eq", "debt_equity", "total_debt_equity", "debt_to_equity"],
    "gross_margin": ["gross_margin"],
    "operating_margin": ["operating_margin"],
    "profit_margin": ["profit_margin"],
    "relative_volume": ["relative_volume", "rel_volume", "rel_vol", "rvol"],
    "average_volume": ["average_volume", "avg_volume", "avg_vol"],
    "volume": ["volume", "current_volume"],
    "price": ["price", "last_price"],
    "change": ["change", "change_percent"],
    "atr": ["average_true_range", "atr"],
    "beta": ["beta"],
    "volatility_week": ["volatility_week", "vol_week"],
    "volatility_month": ["volatility_month", "vol_month"],
    "sma20": ["20_day_simple_moving_average", "sma20", "20_day_sma"],
    "sma50": ["50_day_simple_moving_average", "sma50", "50_day_sma"],
    "sma200": ["200_day_simple_moving_average", "sma200", "200_day_sma"],
    "high52w": ["52w_high", "52_week_high"],
    "low52w": ["52w_low", "52_week_low"],
    "rsi": ["relative_strength_index_14", "rsi", "relative_strength_index"],
    "change_from_open": ["change_from_open"],
    "gap": ["gap"],
    "earnings_date": ["earnings_date", "earnings"],
    "trades": ["trades"],
    "after_hours_volume": ["after_hours_volume", "afterhours_volume"],
}

NUMERIC_FIELDS = {
    "market_cap",
    "pe",
    "forward_pe",
    "peg",
    "ps",
    "pb",
    "pc",
    "pfcf",
    "shares_outstanding",
    "shares_float",
    "float_percent",
    "short_float",
    "short_ratio",
    "short_interest",
    "roa",
    "roe",
    "roi",
    "current_ratio",
    "quick_ratio",
    "lt_debt_to_equity",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
    "change",
    "atr",
    "beta",
    "volatility_week",
    "volatility_month",
    "sma20",
    "sma50",
    "sma200",
    "high52w",
    "low52w",
    "rsi",
    "change_from_open",
    "gap",
    "trades",
    "after_hours_volume",
}

IMPLIED_UNIT_MULTIPLIERS = {
    "market_cap": 1_000_000,
    "shares_outstanding": 1_000_000,
    "shares_float": 1_000_000,
    "short_interest": 1_000_000,
    "average_volume": 1_000,
}


def get_finviz_auth_token() -> str | None:
    try:
        import streamlit as st

        token = st.secrets.get("FINVIZ_AUTH_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass

    return os.getenv("FINVIZ_AUTH_TOKEN")


def configured_finviz_columns() -> str | None:
    configured_columns = [str(value) for value in FINVIZ_COLUMNS.values() if value is not None]
    return ",".join(configured_columns) if configured_columns else None


def build_finviz_export_url(
    columns: str | None = None,
    filters: str | None = None,
    ticker: str | None = None,
) -> tuple[str, dict]:
    params = {
        "v": FINVIZ_CONFIG["view"],
        "ft": FINVIZ_CONFIG["filter_type"],
    }
    if ticker:
        params["t"] = ticker.upper()
    requested_filters = FINVIZ_CONFIG["default_filters"] if filters is None else filters
    if requested_filters:
        params["f"] = requested_filters
    requested_columns = columns or configured_finviz_columns()
    if requested_columns:
        params["c"] = requested_columns
    return FINVIZ_CONFIG["base_url"], params


def build_finviz_preview_url(
    columns: str | None = None,
    filters: str | None = None,
    ticker: str | None = None,
) -> str:
    base_url, params = build_finviz_export_url(columns=columns, filters=filters, ticker=ticker)
    return requests.Request("GET", base_url, params=params).prepare().url or base_url


def fetch_finviz_export(
    columns: str | None = None,
    filters: str | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    token = get_finviz_auth_token()
    if not token:
        raise ValueError("FINVIZ_AUTH_TOKEN is not configured.")

    base_url, params = build_finviz_export_url(columns=columns, filters=filters, ticker=ticker)
    params["auth"] = token
    response = requests.get(
        base_url,
        params=params,
        timeout=FINVIZ_CONFIG["timeout"],
        headers={"User-Agent": "MR-1-Lite/1.0"},
    )
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise ValueError("Finviz returned an empty response.")
    lower_text = text[:500].lower()
    if "<html" in lower_text or "login" in lower_text:
        raise ValueError("Finviz returned HTML/login page instead of CSV. Check FINVIZ_AUTH_TOKEN.")

    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise ValueError("Finviz returned an empty CSV.")
    return frame


@lru_cache(maxsize=16)
def _cached_normalized_export(columns: str | None, filters: str, ticker: str | None) -> pd.DataFrame:
    return normalize_finviz_dataframe(fetch_finviz_export(columns=columns, filters=filters, ticker=ticker))


def fetch_finviz_ticker_snapshot(ticker: str) -> dict:
    symbol = ticker.upper()
    try:
        columns = configured_finviz_columns()
        filters = FINVIZ_CONFIG["default_filters"]
        normalized = _cached_normalized_export(columns, filters, symbol)
        if "ticker" not in normalized.columns:
            return _unavailable("Finviz CSV did not include a ticker column.")

        rows = normalized[normalized["ticker"].astype(str).str.upper() == symbol]
        if rows.empty:
            normalized = _cached_normalized_export(columns, filters, None)
            rows = normalized[normalized["ticker"].astype(str).str.upper() == symbol]
        if rows.empty:
            return _unavailable("Ticker not found in Finviz export.")

        row = rows.iloc[0].to_dict()
        result = {field: row.get(field) for field in NORMALIZED_FIELDS}
        result.update(
            {
                "available": True,
                "source": "finviz",
                "error": None,
                "debug": _debug_payload(normalized, preview_ticker=symbol, columns=columns) if DEBUG_MODE else {},
            }
        )
        return result
    except Exception as exc:
        return _unavailable(_clean_error(exc))


def fetch_finviz_schema_discovery(ticker: str) -> dict:
    symbol = ticker.upper()
    try:
        frame = fetch_finviz_export(columns=FINVIZ_DISCOVERY_COLUMNS, ticker=symbol)
        return {
            "available": True,
            "preview_url": build_finviz_preview_url(columns=FINVIZ_DISCOVERY_COLUMNS, ticker=symbol),
            "headers": list(frame.columns),
            "row_count": int(frame.shape[0]),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "preview_url": build_finviz_preview_url(columns=FINVIZ_DISCOVERY_COLUMNS, ticker=symbol),
            "headers": [],
            "row_count": 0,
            "error": _clean_error(exc),
        }


def normalize_finviz_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [_normalize_column_name(column) for column in normalized.columns]

    resolved_columns = {}
    for target_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized.columns:
                resolved_columns[target_field] = alias
                break

    output = pd.DataFrame(index=normalized.index)
    for field in NORMALIZED_FIELDS:
        source_column = resolved_columns.get(field)
        if source_column:
            output[field] = normalized[source_column]
        else:
            output[field] = pd.NA

    if "ticker" in output.columns:
        output["ticker"] = output["ticker"].astype(str).str.upper()

    for field in NUMERIC_FIELDS:
        if field in output.columns:
            output[field] = output[field].map(lambda value, field=field: parse_human_number(value, field=field))
            source_column = resolved_columns.get(field)
            if field in IMPLIED_UNIT_MULTIPLIERS and source_column and not _column_has_explicit_unit(normalized[source_column]):
                output[field] = output[field].map(lambda value, field=field: _apply_implied_unit(value, field))

    return output


def parse_human_number(value, field: str | None = None):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return _normalize_numeric_value(float(value), field=field)

    text = str(value).strip()
    if not text or text in {"-", "N/A", "nan", "None"}:
        return None

    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
        multiplier = 0.01
    elif text[-1:].upper() in {"K", "M", "B", "T"}:
        suffix = text[-1].upper()
        text = text[:-1]
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}[suffix]

    text = text.replace(",", "")
    try:
        return _normalize_numeric_value(float(text) * multiplier, field=field)
    except ValueError:
        return None


def _normalize_numeric_value(value: float, field: str | None = None) -> float:
    return value


def _apply_implied_unit(value, field: str):
    if value is None or pd.isna(value):
        return None
    number = float(value)
    multiplier = IMPLIED_UNIT_MULTIPLIERS.get(field)
    if not multiplier:
        return number
    if field == "average_volume" and abs(number) >= 10_000_000:
        return number
    if field != "average_volume" and abs(number) >= 1_000_000_000:
        return number
    return number * multiplier


def _normalize_column_name(column: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())
    return normalized.strip("_")


def _column_has_explicit_unit(values: pd.Series) -> bool:
    sample = values.dropna().astype(str).head(50)
    return sample.str.contains(r"[KMBT%]$", case=False, regex=True).any()


def _unavailable(error: str) -> dict:
    result = {field: None for field in NORMALIZED_FIELDS}
    result.update(
        {
            "available": False,
            "source": "finviz",
            "error": error,
            "debug": {"error": error} if DEBUG_MODE else {},
        }
    )
    return result


def _clean_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if "FINVIZ_AUTH_TOKEN" in message:
        return "FINVIZ_AUTH_TOKEN missing or invalid."
    if "login" in message.lower() or "html" in message.lower():
        return "Finviz token missing, expired, or invalid."
    return message


def _debug_payload(frame: pd.DataFrame, preview_ticker: str | None = None, columns: str | None = None) -> dict:
    missing_fields = [field for field in NORMALIZED_FIELDS if field not in frame.columns or frame[field].isna().all()]
    return {
        "columns": list(frame.columns),
        "missing_fields": missing_fields,
        "row_count": int(frame.shape[0]),
        "preview_url": build_finviz_preview_url(columns=columns, ticker=preview_ticker),
    }
