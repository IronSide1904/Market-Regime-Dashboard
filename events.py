from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class EventContextResult:
    available: bool
    catalyst_status: str = "No event data available"
    latest_headline: str = "N/A"
    latest_link: str | None = None
    event_category: str = "Unknown"
    sentiment: str = "Unknown"
    news_count_24h: int = 0
    news_count_7d: int = 0
    filing_count_7d: int | None = None
    insider_count_30d: int | None = None
    earnings_date: str | None = None
    days_to_earnings: int | None = None
    event_risk: str = "Unknown"
    explanation: str = "No event data available."
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def build_event_context(
    ticker: str,
    latest_rvol: float | None,
    volume_context: str,
    options_context: str,
) -> EventContextResult:
    try:
        ticker_obj = yf.Ticker(ticker)
        news_table = _news_table(ticker_obj)
        earnings_date, days_to_earnings = _earnings_context(ticker_obj)
        return classify_event_context(
            news_table=news_table,
            latest_rvol=latest_rvol,
            volume_context=volume_context,
            options_context=options_context,
            earnings_date=earnings_date,
            days_to_earnings=days_to_earnings,
        )
    except Exception as exc:
        return EventContextResult(
            available=False,
            warnings=[_clean_error(exc)],
            explanation="No event data available; volume and options reads are shown without catalyst attribution.",
        )


def classify_event_context(
    news_table: pd.DataFrame,
    latest_rvol: float | None,
    volume_context: str,
    options_context: str,
    earnings_date: str | None,
    days_to_earnings: int | None,
) -> EventContextResult:
    now = pd.Timestamp.now(tz="UTC")
    if news_table.empty:
        news_24h = 0
        news_7d = 0
    else:
        news_24h = int((news_table["Published"] >= now - pd.Timedelta(hours=24)).sum())
        news_7d = int((news_table["Published"] >= now - pd.Timedelta(days=7)).sum())

    has_event_soon = days_to_earnings is not None and 0 <= days_to_earnings <= 7
    elevated_volume = latest_rvol is not None and latest_rvol >= 1.5
    latest = news_table.iloc[0].to_dict() if not news_table.empty else {}
    latest_headline = str(latest.get("Headline", "N/A"))
    category = str(latest.get("Category", "Unknown"))
    sentiment = str(latest.get("Sentiment", "Unknown"))

    if elevated_volume and news_24h > 0:
        catalyst_status = "News-confirmed volume"
    elif elevated_volume and has_event_soon:
        catalyst_status = "Earnings-related"
    elif elevated_volume:
        catalyst_status = "Unexplained volume"
    elif has_event_soon:
        catalyst_status = "Earnings watch"
    elif news_7d > 0:
        catalyst_status = "Recent news"
    else:
        catalyst_status = "No unusual volume"

    event_risk = "Elevated" if has_event_soon or "Event Risk" in options_context or catalyst_status in {"Unexplained volume", "News-confirmed volume"} else "Normal"
    explanation = _event_explanation(
        catalyst_status=catalyst_status,
        latest_headline=latest_headline,
        days_to_earnings=days_to_earnings,
        latest_rvol=latest_rvol,
        volume_context=volume_context,
        options_context=options_context,
    )

    return EventContextResult(
        available=not news_table.empty or earnings_date is not None,
        catalyst_status=catalyst_status,
        latest_headline=latest_headline,
        latest_link=latest.get("Link"),
        event_category=category,
        sentiment=sentiment,
        news_count_24h=news_24h,
        news_count_7d=news_7d,
        earnings_date=earnings_date,
        days_to_earnings=days_to_earnings,
        event_risk=event_risk,
        explanation=explanation,
        table=news_table,
    )


def _news_table(ticker_obj: yf.Ticker) -> pd.DataFrame:
    raw_news = getattr(ticker_obj, "news", []) or []
    rows = []
    for item in raw_news[:25]:
        parsed = _parse_news_item(item)
        if parsed:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame(columns=["Published", "Source", "Category", "Sentiment", "Headline", "Link"])
    table = pd.DataFrame(rows)
    table = table.sort_values("Published", ascending=False)
    return table


def _parse_news_item(item: dict) -> dict | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    headline = content.get("title") or item.get("title")
    if not headline:
        return None
    timestamp = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    published = _parse_timestamp(timestamp)
    provider = content.get("provider")
    source = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher") or item.get("publisher")
    canonical_url = content.get("canonicalUrl")
    link = canonical_url.get("url") if isinstance(canonical_url, dict) else content.get("link") or item.get("link")
    return {
        "Published": published,
        "Source": source,
        "Category": _categorize_headline(headline),
        "Sentiment": _sentiment(headline),
        "Headline": headline,
        "Link": link,
    }


def _earnings_context(ticker_obj: yf.Ticker) -> tuple[str | None, int | None]:
    today = pd.Timestamp.today().normalize()
    candidates: list[pd.Timestamp] = []
    try:
        dates = ticker_obj.get_earnings_dates(limit=8)
        if dates is not None and not dates.empty:
            for idx in dates.index:
                candidate = pd.Timestamp(idx).tz_localize(None).normalize()
                if candidate >= today:
                    candidates.append(candidate)
    except Exception:
        pass
    try:
        calendar = ticker_obj.calendar
        if isinstance(calendar, pd.DataFrame) and "Earnings Date" in calendar.index:
            value = calendar.loc["Earnings Date"].dropna().iloc[0]
            candidates.append(pd.Timestamp(value).tz_localize(None).normalize())
        elif isinstance(calendar, dict):
            value = calendar.get("Earnings Date") or calendar.get("earningsDate")
            if isinstance(value, list):
                value = value[0] if value else None
            if value:
                candidates.append(pd.Timestamp(value).tz_localize(None).normalize())
    except Exception:
        pass
    future = sorted({candidate for candidate in candidates if candidate >= today})
    if not future:
        return None, None
    earnings = future[0]
    return earnings.strftime("%Y-%m-%d"), int((earnings - today).days)


def _parse_timestamp(value) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp(datetime.now(timezone.utc))
    return parsed


def _categorize_headline(headline: str) -> str:
    lower = headline.lower()
    rules = [
        ("Earnings", ["earnings", "eps", "revenue", "quarter"]),
        ("Guidance", ["guidance", "outlook", "forecast"]),
        ("Analyst upgrade", ["upgrade", "raises rating"]),
        ("Analyst downgrade", ["downgrade", "cuts rating"]),
        ("Price target change", ["price target", "pt "]),
        ("Contract / partnership", ["contract", "order", "partnership", "deal"]),
        ("Product launch", ["launch", "unveils", "introduces"]),
        ("M&A", ["acquire", "acquisition", "merger", "buyout"]),
        ("SEC filing", ["filing", "sec", "10-k", "10-q", "8-k"]),
        ("Regulatory / lawsuit", ["lawsuit", "regulatory", "probe", "investigation"]),
        ("Macro event", ["fed", "inflation", "jobs", "rates", "tariff"]),
    ]
    for category, keywords in rules:
        if any(keyword in lower for keyword in keywords):
            return category
    return "Market news"


def _sentiment(headline: str) -> str:
    lower = headline.lower()
    positive = ["beats", "raises", "upgrade", "wins", "record", "growth", "approval", "surges"]
    negative = ["misses", "cuts", "downgrade", "lawsuit", "probe", "falls", "warning", "loss"]
    pos = any(word in lower for word in positive)
    neg = any(word in lower for word in negative)
    if pos and neg:
        return "Mixed"
    if pos:
        return "Positive"
    if neg:
        return "Negative"
    return "Unknown"


def _event_explanation(
    catalyst_status: str,
    latest_headline: str,
    days_to_earnings: int | None,
    latest_rvol: float | None,
    volume_context: str,
    options_context: str,
) -> str:
    rvol_text = "RVOL unavailable" if latest_rvol is None else f"RVOL is {latest_rvol:.2f}x"
    if catalyst_status == "News-confirmed volume":
        return f"Volume is elevated and news-confirmed by a recent headline: {latest_headline}"
    if catalyst_status == "Earnings-related":
        return f"{rvol_text} and earnings are within {days_to_earnings} days, so treat the move as event-sensitive."
    if catalyst_status == "Unexplained volume":
        return f"{rvol_text}, but no obvious catalyst was found. Treat {volume_context.lower()} as unexplained flow."
    if "Event Risk" in options_context:
        return "Options IV is elevated into a possible event window. Reduce size unless the event risk is intentional."
    if catalyst_status == "Recent news":
        return f"Recent news exists, but volume is not unusually high. Latest headline: {latest_headline}"
    return "No clear catalyst is attached to the latest volume/regime read."


def _clean_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return text.replace("\n", " ")[:180]
