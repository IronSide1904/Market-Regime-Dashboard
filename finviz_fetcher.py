from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DEBUG_MODE, FINVIZ_COLUMNS, FINVIZ_CONFIG


load_dotenv()
load_dotenv(Path(__file__).with_name("Finviz API Token.env"), override=False)

NORMALIZED_FIELDS = [
    "ticker",
    "company",
    "sector",
    "industry",
    "market_cap",
    "shares_outstanding",
    "shares_float",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
    "change",
    "atr",
    "beta",
    "volatility_week",
    "volatility_month",
    "change_from_open",
    "gap",
    "short_float",
    "short_interest",
    "float_percent",
    "trades",
    "after_hours_volume",
]

COLUMN_ALIASES = {
    "ticker": ["ticker", "symbol"],
    "company": ["company", "company_name", "name"],
    "sector": ["sector"],
    "industry": ["industry"],
    "market_cap": ["market_cap", "marketcap", "market_capitalization"],
    "shares_outstanding": ["shares_outstanding", "shs_outstand", "shs_outstanding", "shares_out"],
    "shares_float": ["shares_float", "shs_float", "float_shares"],
    "relative_volume": ["relative_volume", "rel_volume", "rel_vol", "rvol"],
    "average_volume": ["average_volume", "avg_volume", "avg_vol"],
    "volume": ["volume", "current_volume"],
    "price": ["price", "last_price"],
    "change": ["change", "change_percent"],
    "atr": ["atr", "average_true_range"],
    "beta": ["beta"],
    "volatility_week": ["volatility_week", "vol_week"],
    "volatility_month": ["volatility_month", "vol_month"],
    "change_from_open": ["change_from_open"],
    "gap": ["gap"],
    "short_float": ["short_float", "short_float_percent", "short_float_pct"],
    "short_interest": ["short_interest", "shares_short"],
    "float_percent": ["float", "float_percent", "float_outstanding"],
    "trades": ["trades"],
    "after_hours_volume": ["after_hours_volume", "afterhours_volume"],
}

NUMERIC_FIELDS = {
    "market_cap",
    "shares_outstanding",
    "shares_float",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
    "change",
    "atr",
    "beta",
    "volatility_week",
    "volatility_month",
    "change_from_open",
    "gap",
    "short_float",
    "short_interest",
    "float_percent",
    "trades",
    "after_hours_volume",
}

FINVIZ_UNIT_MULTIPLIERS = {
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
    configured_columns = [str(value) for value in FINVIZ_COLUMNS.values() if value]
    return ",".join(configured_columns) if configured_columns else None


def build_finviz_export_url(columns: str | None = None, filters: str | None = None) -> tuple[str, dict]:
    params = {
        "v": FINVIZ_CONFIG["view"],
        "ft": FINVIZ_CONFIG["filter_type"],
    }
    requested_filters = FINVIZ_CONFIG["default_filters"] if filters is None else filters
    if requested_filters:
        params["f"] = requested_filters
    requested_columns = columns or configured_finviz_columns()
    if requested_columns:
        params["c"] = requested_columns
    return FINVIZ_CONFIG["base_url"], params


def fetch_finviz_export(columns: str | None = None, filters: str | None = None) -> pd.DataFrame:
    token = get_finviz_auth_token()
    if not token:
        raise ValueError("FINVIZ_AUTH_TOKEN is not configured.")

    base_url, params = build_finviz_export_url(columns=columns, filters=filters)
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
def _cached_normalized_export(columns: str | None, filters: str) -> pd.DataFrame:
    return normalize_finviz_dataframe(fetch_finviz_export(columns=columns, filters=filters))


def fetch_finviz_ticker_snapshot(ticker: str) -> dict:
    symbol = ticker.upper()
    try:
        normalized = _cached_normalized_export(configured_finviz_columns(), FINVIZ_CONFIG["default_filters"])
        if "ticker" not in normalized.columns:
            return _unavailable("Finviz CSV did not include a ticker column.")

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
                "debug": _debug_payload(normalized) if DEBUG_MODE else {},
            }
        )
        return result
    except Exception as exc:
        return _unavailable(_clean_error(exc))


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
            output[field] = output[field].map(parse_human_number)
            source_column = resolved_columns.get(field)
            if field in FINVIZ_UNIT_MULTIPLIERS and source_column and not _column_has_explicit_unit(normalized[source_column]):
                output[field] = output[field].map(
                    lambda value: value * FINVIZ_UNIT_MULTIPLIERS[field] if value is not None else None
                )

    return output


def parse_human_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

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
        return float(text) * multiplier
    except ValueError:
        return None


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


def _debug_payload(frame: pd.DataFrame) -> dict:
    missing_fields = [field for field in NORMALIZED_FIELDS if field not in frame.columns or frame[field].isna().all()]
    return {
        "columns": list(frame.columns),
        "missing_fields": missing_fields,
        "row_count": int(frame.shape[0]),
    }
