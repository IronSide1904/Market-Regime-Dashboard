from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math

import numpy as np
import pandas as pd
import yfinance as yf

MIN_REASONABLE_IV = 0.03
MAX_REASONABLE_IV = 5.0
DEFAULT_RISK_FREE_RATE = 0.045


@dataclass(frozen=True)
class OptionsVolatilityResult:
    available: bool
    source: str
    context: str
    current_iv: float | None = None
    iv_rank: float | None = None
    iv_percentile: float | None = None
    finviz_volatility_week: float | None = None
    finviz_volatility_month: float | None = None
    realized_volatility_20d: float | None = None
    iv_premium: float | None = None
    put_call_volume_ratio: float | None = None
    put_call_oi_ratio: float | None = None
    options_volume_oi_ratio: float | None = None
    nearest_expiration: str | None = None
    days_to_expiration: int | None = None
    expirations_analyzed: int = 0
    atm_call_iv: float | None = None
    atm_put_iv: float | None = None
    skew: float | None = None
    expiration_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    interpretation: str = "Options volatility data is unavailable."
    warnings: list[str] = field(default_factory=list)


UNAVAILABLE_OPTIONS = OptionsVolatilityResult(
    available=False,
    source="unavailable",
    context="Unavailable",
    warnings=["Options chain data is unavailable for this ticker."],
)


def build_options_volatility_context(
    ticker: str,
    price_df: pd.DataFrame,
    latest_regime: str,
    latest_rvol: float | None = None,
    max_expirations: int = 6,
    finviz_snapshot: dict | None = None,
) -> OptionsVolatilityResult:
    realized_volatility = calculate_realized_volatility(price_df)
    finviz_week = _valid_float((finviz_snapshot or {}).get("volatility_week"))
    finviz_month = _valid_float((finviz_snapshot or {}).get("volatility_month"))
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = _valid_expirations(ticker_obj)
        if not expirations:
            return _unavailable_with_realized(
                realized_volatility,
                "No listed options expiration was found.",
                finviz_week,
                finviz_month,
            )

        spot = _latest_price(price_df)
        if spot is None:
            return _unavailable_with_realized(
                realized_volatility,
                "Could not infer the current underlying price.",
                finviz_week,
                finviz_month,
            )

        expiration_rows = _expiration_summaries(ticker_obj=ticker_obj, expirations=expirations[:max_expirations], spot=spot)
        if expiration_rows.empty:
            return _unavailable_with_realized(
                realized_volatility,
                "Usable option chains were not found across listed expirations.",
                finviz_week,
                finviz_month,
            )
        if realized_volatility is not None and "ATM Blended IV" in expiration_rows.columns:
            expiration_rows["IV Premium"] = expiration_rows["ATM Blended IV"] - realized_volatility

        usable_rows = expiration_rows.dropna(subset=["ATM Blended IV"])
        if usable_rows.empty:
            return _unavailable_with_realized(
                realized_volatility,
                "ATM implied volatility is unavailable across listed expirations.",
                finviz_week,
                finviz_month,
            )

        primary_candidates = usable_rows[usable_rows["DTE"] >= 1]
        primary = (primary_candidates if not primary_candidates.empty else usable_rows).iloc[0]
        expiration = str(primary["Expiration"])
        current_iv = float(primary["ATM Blended IV"])
        atm_call_iv = _valid_float(primary["ATM Call IV"])
        atm_put_iv = _valid_float(primary["ATM Put IV"])
        put_call_volume_ratio = _valid_float(primary["Put/Call Vol"])
        put_call_oi_ratio = _valid_float(primary["Put/Call OI"])
        options_volume_oi_ratio = _valid_float(primary["Volume/OI"])
        dte = int(primary["DTE"])
        iv_premium = None if realized_volatility is None else current_iv - realized_volatility
        skew = None if atm_call_iv is None or atm_put_iv is None else atm_put_iv - atm_call_iv
        context = classify_options_context(
            current_iv=current_iv,
            realized_volatility=realized_volatility,
            put_call_volume_ratio=put_call_volume_ratio,
            latest_regime=latest_regime,
            latest_rvol=latest_rvol,
            days_to_expiration=dte,
        )

        return OptionsVolatilityResult(
            available=True,
            source="yfinance + finviz" if finviz_week is not None or finviz_month is not None else "yfinance",
            context=context,
            current_iv=current_iv,
            finviz_volatility_week=finviz_week,
            finviz_volatility_month=finviz_month,
            realized_volatility_20d=realized_volatility,
            iv_premium=iv_premium,
            put_call_volume_ratio=put_call_volume_ratio,
            put_call_oi_ratio=put_call_oi_ratio,
            options_volume_oi_ratio=options_volume_oi_ratio,
            nearest_expiration=expiration,
            days_to_expiration=dte,
            expirations_analyzed=int(len(expiration_rows.index)),
            atm_call_iv=atm_call_iv,
            atm_put_iv=atm_put_iv,
            skew=skew,
            expiration_table=expiration_rows,
            interpretation=_options_interpretation(
                context=context,
                latest_regime=latest_regime,
                current_iv=current_iv,
                realized_volatility=realized_volatility,
                iv_premium=iv_premium,
                put_call_volume_ratio=put_call_volume_ratio,
                finviz_week=finviz_week,
                finviz_month=finviz_month,
                latest_rvol=latest_rvol,
            ),
            warnings=_options_warnings(
                expiration_rows=expiration_rows,
                primary_expiration=expiration,
                finviz_week=finviz_week,
                finviz_month=finviz_month,
            ),
        )
    except Exception as exc:
        return _unavailable_with_realized(realized_volatility, _clean_error(exc), finviz_week, finviz_month)


def calculate_realized_volatility(price_df: pd.DataFrame, window: int = 20) -> float | None:
    if price_df.empty or "Close" not in price_df.columns:
        return None
    close = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    if close.shape[0] <= window:
        return None
    returns = np.log(close / close.shift(1)).dropna()
    realized = returns.tail(window).std() * np.sqrt(252)
    return _valid_float(realized)


def classify_options_context(
    current_iv: float,
    realized_volatility: float | None,
    put_call_volume_ratio: float | None,
    latest_regime: str,
    latest_rvol: float | None,
    days_to_expiration: int | None,
) -> str:
    premium = None if realized_volatility is None else current_iv - realized_volatility
    if premium is not None and premium >= 0.25 and days_to_expiration is not None and days_to_expiration <= 14:
        return "IV Event Risk"
    if current_iv >= 0.75 or (premium is not None and premium >= 0.35):
        return "IV Stress"
    if latest_regime == "Defensive" and (put_call_volume_ratio or 0) >= 1.5 and current_iv >= 0.45:
        return "IV Stress"
    if current_iv <= 0.25 and latest_regime == "Risk-On" and latest_rvol is not None and latest_rvol >= 1.2:
        return "IV Compression / Squeeze Candidate"
    if premium is not None and premium >= 0.15:
        return "IV Elevated"
    if current_iv <= 0.25:
        return "IV Calm"
    return "IV Normal"


def _valid_expirations(ticker_obj: yf.Ticker) -> list[str]:
    expirations = list(getattr(ticker_obj, "options", []) or [])
    today = pd.Timestamp.today().normalize()
    return [expiration for expiration in expirations if pd.Timestamp(expiration) >= today]


def _expiration_summaries(ticker_obj: yf.Ticker, expirations: list[str], spot: float) -> pd.DataFrame:
    rows = []
    for expiration in expirations:
        dte = max((pd.Timestamp(expiration).date() - date.today()).days, 0)
        try:
            chain = ticker_obj.option_chain(expiration)
        except Exception:
            continue
        calls = _clean_options_frame(chain.calls, option_type="call", spot=spot, dte=dte)
        puts = _clean_options_frame(chain.puts, option_type="put", spot=spot, dte=dte)
        if calls.empty or puts.empty:
            continue
        atm_call_iv = _atm_iv(calls, spot)
        atm_put_iv = _atm_iv(puts, spot)
        blended_iv = _mean_valid([atm_call_iv, atm_put_iv])
        iv_source = _iv_source_label(calls=calls, puts=puts, atm_call_iv=atm_call_iv, atm_put_iv=atm_put_iv)
        total_call_volume = calls["volume"].sum()
        total_put_volume = puts["volume"].sum()
        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()
        total_volume = total_call_volume + total_put_volume
        total_oi = total_call_oi + total_put_oi
        rows.append(
            {
                "Expiration": expiration,
                "DTE": dte,
                "ATM Call IV": atm_call_iv,
                "ATM Put IV": atm_put_iv,
                "ATM Blended IV": blended_iv,
                "IV Source": iv_source,
                "IV Premium": np.nan,
                "Put/Call Vol": _safe_ratio(total_put_volume, total_call_volume),
                "Put/Call OI": _safe_ratio(total_put_oi, total_call_oi),
                "Options Volume": total_volume,
                "Open Interest": total_oi,
                "Volume/OI": _safe_ratio(total_volume, total_oi),
            }
        )
    return pd.DataFrame(rows).sort_values("DTE") if rows else pd.DataFrame()


def _clean_options_frame(frame: pd.DataFrame, option_type: str, spot: float, dte: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    cleaned = frame.copy()
    for column in ["strike", "impliedVolatility", "volume", "openInterest", "lastPrice", "bid", "ask"]:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["volume"] = cleaned["volume"].fillna(0)
    cleaned["openInterest"] = cleaned["openInterest"].fillna(0)
    cleaned = cleaned[cleaned["strike"].notna()].copy()
    cleaned["reportedIv"] = cleaned["impliedVolatility"].apply(_reasonable_iv)
    cleaned["derivedIv"] = cleaned.apply(
        lambda row: _derive_implied_volatility(
            option_type=option_type,
            spot=spot,
            strike=row["strike"],
            dte=dte,
            bid=row["bid"],
            ask=row["ask"],
            last_price=row["lastPrice"],
        ),
        axis=1,
    )
    cleaned["usableIv"] = cleaned["reportedIv"].combine_first(cleaned["derivedIv"])
    cleaned["ivSource"] = np.where(
        cleaned["reportedIv"].notna(),
        "reported",
        np.where(cleaned["derivedIv"].notna(), "price-derived", "unavailable"),
    )
    return cleaned


def _atm_iv(frame: pd.DataFrame, spot: float) -> float | None:
    if frame.empty:
        return None
    iv_frame = frame[frame["usableIv"].notna()].copy()
    if iv_frame.empty:
        return None
    liquid = iv_frame[(iv_frame["volume"] > 0) | (iv_frame["openInterest"] > 0)].copy()
    candidates = liquid if not liquid.empty else iv_frame.copy()
    candidates["distance"] = (candidates["strike"] - spot).abs()
    value = candidates.sort_values(["distance", "openInterest", "volume"], ascending=[True, False, False]).iloc[0][
        "usableIv"
    ]
    return _valid_float(value)


def _iv_source_label(calls: pd.DataFrame, puts: pd.DataFrame, atm_call_iv: float | None, atm_put_iv: float | None) -> str:
    sources = []
    for frame, value in [(calls, atm_call_iv), (puts, atm_put_iv)]:
        if value is None or frame.empty:
            continue
        match = frame[np.isclose(frame["usableIv"], value, rtol=0, atol=1e-8)]
        if not match.empty:
            sources.append(str(match.iloc[0]["ivSource"]))
    if not sources:
        return "Unavailable"
    if all(source == "reported" for source in sources):
        return "Reported"
    if all(source == "price-derived" for source in sources):
        return "Price-derived"
    return "Mixed"


def _reasonable_iv(value) -> float | None:
    number = _valid_float(value)
    if number is None:
        return None
    if number < MIN_REASONABLE_IV or number > MAX_REASONABLE_IV:
        return None
    return number


def _derive_implied_volatility(
    option_type: str,
    spot: float,
    strike: float,
    dte: int,
    bid: float | None,
    ask: float | None,
    last_price: float | None,
) -> float | None:
    mark = _option_mark(bid=bid, ask=ask, last_price=last_price)
    spot = _valid_float(spot)
    strike = _valid_float(strike)
    if mark is None or spot is None or strike is None or spot <= 0 or strike <= 0:
        return None

    years = max(float(dte), 1.0) / 365.0
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if mark <= intrinsic:
        return None

    low = MIN_REASONABLE_IV
    high = MAX_REASONABLE_IV
    low_price = _black_scholes_price(option_type, spot, strike, years, DEFAULT_RISK_FREE_RATE, low)
    high_price = _black_scholes_price(option_type, spot, strike, years, DEFAULT_RISK_FREE_RATE, high)
    if low_price is None or high_price is None or mark < low_price or mark > high_price:
        return None

    for _ in range(80):
        mid = (low + high) / 2
        price = _black_scholes_price(option_type, spot, strike, years, DEFAULT_RISK_FREE_RATE, mid)
        if price is None:
            return None
        if abs(price - mark) < 0.0001:
            return _reasonable_iv(mid)
        if price < mark:
            low = mid
        else:
            high = mid
    return _reasonable_iv((low + high) / 2)


def _option_mark(bid: float | None, ask: float | None, last_price: float | None) -> float | None:
    bid_value = _valid_float(bid)
    ask_value = _valid_float(ask)
    last_value = _valid_float(last_price)
    if bid_value is not None and ask_value is not None and bid_value > 0 and ask_value > 0 and ask_value >= bid_value:
        return (bid_value + ask_value) / 2
    if last_value is not None and last_value > 0:
        return last_value
    return None


def _black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    years: float,
    risk_free_rate: float,
    volatility: float,
) -> float | None:
    if spot <= 0 or strike <= 0 or years <= 0 or volatility <= 0:
        return None
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * years) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * math.exp(-risk_free_rate * years) * _normal_cdf(d2)
    return strike * math.exp(-risk_free_rate * years) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _latest_price(price_df: pd.DataFrame) -> float | None:
    if price_df.empty or "Close" not in price_df.columns:
        return None
    close = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    return _valid_float(close.iloc[-1]) if not close.empty else None


def _chain_spot(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    strikes = pd.concat([calls["strike"], puts["strike"]], ignore_index=True).dropna()
    return _valid_float(strikes.median()) if not strikes.empty else None


def _options_interpretation(
    context: str,
    latest_regime: str,
    current_iv: float,
    realized_volatility: float | None,
    iv_premium: float | None,
    put_call_volume_ratio: float | None,
    finviz_week: float | None,
    finviz_month: float | None,
    latest_rvol: float | None,
) -> str:
    iv_text = f"current ATM IV is {current_iv:.1%}"
    realized_text = "20D realized volatility is unavailable" if realized_volatility is None else f"20D realized volatility is {realized_volatility:.1%}"
    premium_text = "" if iv_premium is None else f", leaving an IV premium of {iv_premium:+.1%}"
    put_call_text = (
        "put/call volume is unavailable"
        if put_call_volume_ratio is None
        else f"put/call volume is {put_call_volume_ratio:.2f}"
    )
    finviz_text = _finviz_volatility_text(finviz_week=finviz_week, finviz_month=finviz_month)
    if context in {"IV Event Risk", "IV Stress"}:
        return f"{context}: {iv_text}, {realized_text}{premium_text}, and {put_call_text}. {finviz_text}Reduce size unless event risk is intentional."
    if context == "IV Compression / Squeeze Candidate":
        return f"{context}: {iv_text} while MR-1 is {latest_regime} and RVOL is {latest_rvol:.2f}x. {finviz_text}A volatility expansion setup is possible."
    if context == "IV Elevated":
        return f"{context}: {iv_text}, {realized_text}{premium_text}. {finviz_text}The setup can still work, but options pricing is warning that movement risk is higher."
    return f"{context}: {iv_text}, {realized_text}{premium_text}. {finviz_text}Options are not adding a major warning right now."


def _finviz_volatility_text(finviz_week: float | None, finviz_month: float | None) -> str:
    parts = []
    if finviz_week is not None:
        parts.append(f"Finviz week volatility is {finviz_week:.1%}")
    if finviz_month is not None:
        parts.append(f"month volatility is {finviz_month:.1%}")
    if not parts:
        return ""
    return f"{' and '.join(parts)}. "


def _options_warnings(
    expiration_rows: pd.DataFrame,
    primary_expiration: str,
    finviz_week: float | None,
    finviz_month: float | None,
) -> list[str]:
    warnings = [
        f"Current IV uses expiration {primary_expiration}; showing {len(expiration_rows.index)} listed expiration dates.",
    ]
    if finviz_week is not None or finviz_month is not None:
        warnings.append("Finviz volatility is used as a broad timing/positioning overlay, not as true IV Rank or IV Percentile.")
    if "IV Source" in expiration_rows.columns and expiration_rows["IV Source"].astype(str).str.contains("derived", case=False).any():
        warnings.append(
            "Some yfinance reported IV values looked like placeholders, so IV was derived from bid/ask or last option prices."
        )
    warnings.append("IV Rank and IV Percentile need historical IV data, so they are marked unavailable.")
    return warnings


def _unavailable_with_realized(
    realized_volatility: float | None,
    warning: str,
    finviz_week: float | None = None,
    finviz_month: float | None = None,
) -> OptionsVolatilityResult:
    finviz_text = _finviz_volatility_text(finviz_week=finviz_week, finviz_month=finviz_month)
    return OptionsVolatilityResult(
        available=False,
        source="finviz" if finviz_week is not None or finviz_month is not None else "unavailable",
        context="Unavailable",
        finviz_volatility_week=finviz_week,
        finviz_volatility_month=finviz_month,
        realized_volatility_20d=realized_volatility,
        interpretation=(
            "Options chain data is unavailable; "
            f"{finviz_text}"
            "use VIX, ATR, Finviz volatility, and realized volatility as the fallback volatility read."
        ),
        warnings=[warning],
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator in (0, None) or pd.isna(denominator):
        return None
    return _valid_float(numerator / denominator)


def _mean_valid(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and not pd.isna(value)]
    return _valid_float(np.mean(valid)) if valid else None


def _valid_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or np.isinf(number):
        return None
    return number


def _clean_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return text.replace("\n", " ")[:180]
