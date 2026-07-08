from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yfinance as yf
from yfinance import EquityQuery
from yfinance.const import SECTOR_INDUSTY_MAPPING


@dataclass(frozen=True)
class AssetContext:
    ticker: str
    sector_etf: str
    industry_proxy: str
    peers: list[str]
    theme_ticker: str = ""
    sub_industry_ticker: str = ""
    peer_source: str = "Curated fallback"


ASSET_CONTEXTS = {
    "AAPL": AssetContext("AAPL", "XLK", "VGT", ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "AVGO"], "QQQ", "VGT"),
    "MSFT": AssetContext("MSFT", "XLK", "IGV", ["ORCL", "CRM", "ADBE", "NOW", "GOOGL", "AMZN"], "CLOU", "IGV"),
    "NVDA": AssetContext("NVDA", "XLK", "SMH", ["AMD", "AVGO", "TSM", "ASML", "MU", "MRVL"], "AIQ", "SOXX"),
    "AMD": AssetContext("AMD", "XLK", "SMH", ["NVDA", "AVGO", "INTC", "TSM", "MU", "MRVL"], "AIQ", "SOXX"),
    "TSLA": AssetContext("TSLA", "XLY", "CARZ", ["RIVN", "GM", "F", "LCID", "NIO", "XPEV"], "DRIV", "CARZ"),
    "AMZN": AssetContext("AMZN", "XLY", "FDIS", ["WMT", "COST", "MELI", "BABA", "SHOP", "EBAY"], "IBUY", "ONLN"),
    "META": AssetContext("META", "XLC", "XLC", ["GOOGL", "SNAP", "PINS", "NFLX", "TTD", "RDDT"], "SOCL", "FDN"),
    "GOOGL": AssetContext("GOOGL", "XLC", "XLC", ["META", "MSFT", "AMZN", "NFLX", "TTD", "RDDT"], "SOCL", "FDN"),
    "RKLB": AssetContext("RKLB", "XLI", "ITA", ["ASTS", "LUNR", "RDW", "PL", "SPCE", "IRDM"], "ARKX", "UFO"),
    "SPY": AssetContext("SPY", "SPY", "SPY", ["QQQ", "IWM", "DIA", "RSP", "MDY", "VTI"], "QQQ", "RSP"),
    "QQQ": AssetContext("QQQ", "QQQ", "QQQ", ["SPY", "XLK", "SMH", "IWM", "ARKK", "IGV"], "XLK", "SMH"),
    "BTC-USD": AssetContext("BTC-USD", "QQQ", "BTC-USD", ["ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"], "BLOK", "ETH-USD"),
    "ETH-USD": AssetContext("ETH-USD", "QQQ", "ETH-USD", ["BTC-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"], "BLOK", "SOL-USD"),
}


def _industry_key(industry: str) -> str:
    return (
        industry.replace("—", "-")
        .replace("–", "-")
        .replace(" - ", "-")
        .strip()
    )


SECTOR_ETFS = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

INDUSTRY_PROXIES = {
    "Auto Manufacturers": "CARZ",
    "Consumer Electronics": "VGT",
    "Internet Content & Information": "XLC",
    "Internet Retail": "FDIS",
    "Semiconductors": "SMH",
    "Software-Infrastructure": "IGV",
    "Software-Application": "IGV",
}

THEME_TICKERS = {
    "Auto Manufacturers": "DRIV",
    "Consumer Electronics": "AIQ",
    "Internet Content & Information": "SOCL",
    "Internet Retail": "IBUY",
    "Semiconductors": "AIQ",
    "Software-Infrastructure": "CLOU",
    "Software-Application": "CLOU",
}

SUB_INDUSTRY_TICKERS = {
    "Auto Manufacturers": "CARZ",
    "Consumer Electronics": "VGT",
    "Internet Content & Information": "FDN",
    "Internet Retail": "ONLN",
    "Semiconductors": "SOXX",
    "Software-Infrastructure": "IGV",
    "Software-Application": "IGV",
}

_CANONICAL_INDUSTRIES = {
    _industry_key(industry): industry
    for industries in SECTOR_INDUSTY_MAPPING.values()
    for industry in industries
}


def get_asset_context(ticker: str, benchmark: str) -> AssetContext:
    normalized_ticker = ticker.upper()
    curated = ASSET_CONTEXTS.get(normalized_ticker)
    if curated is not None:
        return AssetContext(
            ticker=curated.ticker,
            sector_etf=curated.sector_etf,
            industry_proxy=curated.industry_proxy,
            peers=curated.peers,
            theme_ticker=curated.theme_ticker,
            sub_industry_ticker=curated.sub_industry_ticker,
            peer_source="Curated exact ticker mapping",
        )

    fallback = AssetContext(
        ticker=normalized_ticker,
        sector_etf=benchmark,
        industry_proxy=benchmark,
        peers=[],
    )

    yahoo_context = _get_yahoo_context(
        ticker=normalized_ticker,
        fallback_sector_etf=fallback.sector_etf,
        fallback_industry_proxy=fallback.industry_proxy,
        fallback_theme_ticker=fallback.theme_ticker,
        fallback_sub_industry_ticker=fallback.sub_industry_ticker,
        fallback_peers=tuple(fallback.peers),
    )
    if yahoo_context is not None:
        return yahoo_context

    return fallback


@lru_cache(maxsize=256)
def _get_yahoo_context(
    ticker: str,
    fallback_sector_etf: str,
    fallback_industry_proxy: str,
    fallback_theme_ticker: str,
    fallback_sub_industry_ticker: str,
    fallback_peers: tuple[str, ...],
) -> AssetContext | None:
    metadata = _lookup_yahoo_metadata(ticker)
    if not metadata:
        return None

    quote_type = str(metadata.get("quoteType") or "").upper()
    if quote_type != "EQUITY":
        return None

    sector = str(metadata.get("sector") or metadata.get("sectorDisp") or "")
    industry = str(metadata.get("industry") or metadata.get("industryDisp") or "")
    canonical_industry = _canonical_industry(industry)
    if not canonical_industry:
        return None

    peers = _screen_yahoo_industry_peers(
        ticker=ticker,
        industry=canonical_industry,
        fallback_peers=list(fallback_peers),
    )
    if not peers:
        return None

    sector_etf = SECTOR_ETFS.get(sector, fallback_sector_etf)
    industry_key = _industry_key(canonical_industry)
    industry_proxy = INDUSTRY_PROXIES.get(industry_key, fallback_industry_proxy)
    if industry_proxy == fallback_industry_proxy and fallback_industry_proxy == ticker:
        industry_proxy = sector_etf
    theme_ticker = THEME_TICKERS.get(industry_key, fallback_theme_ticker)
    sub_industry_ticker = SUB_INDUSTRY_TICKERS.get(industry_key, fallback_sub_industry_ticker)

    return AssetContext(
        ticker=ticker,
        sector_etf=sector_etf,
        industry_proxy=industry_proxy,
        peers=peers,
        theme_ticker=theme_ticker,
        sub_industry_ticker=sub_industry_ticker,
        peer_source=f"Yahoo industry screener: {canonical_industry}",
    )


def _lookup_yahoo_metadata(ticker: str) -> dict:
    try:
        search = yf.Search(ticker, max_results=8, news_count=0, lists_count=0)
    except Exception:
        return {}

    for quote in search.quotes or []:
        if str(quote.get("symbol", "")).upper() == ticker:
            return quote
    return {}


def _screen_yahoo_industry_peers(
    ticker: str,
    industry: str,
    fallback_peers: list[str],
    max_peers: int = 6,
) -> list[str]:
    query = EquityQuery(
        "and",
        [
            EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
            EquityQuery("eq", ["industry", industry]),
            EquityQuery("gte", ["intradaymarketcap", 1_000_000_000]),
        ],
    )

    try:
        response = yf.screen(
            query,
            size=25,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
    except Exception:
        return _dedupe_peers(ticker, fallback_peers, max_peers)

    screened = [
        str(quote.get("symbol", "")).upper()
        for quote in response.get("quotes", [])
        if str(quote.get("quoteType", "")).upper() == "EQUITY"
    ]
    return _dedupe_peers(ticker, [*screened, *fallback_peers], max_peers)


def _dedupe_peers(ticker: str, candidates: list[str], max_peers: int) -> list[str]:
    peers: list[str] = []
    blocked_characters = {"^", "=", "."}
    for candidate in candidates:
        symbol = candidate.strip().upper()
        if not symbol or symbol == ticker or any(character in symbol for character in blocked_characters):
            continue
        if symbol not in peers:
            peers.append(symbol)
        if len(peers) >= max_peers:
            break
    return peers


def _canonical_industry(industry: str) -> str | None:
    return _CANONICAL_INDUSTRIES.get(_industry_key(industry))
