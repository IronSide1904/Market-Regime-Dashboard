from __future__ import annotations

from metadata import get_asset_context


def get_peer_tickers(ticker: str, benchmark: str) -> list[str]:
    context = get_asset_context(ticker=ticker, benchmark=benchmark)
    return [peer for peer in context.peers if peer != ticker.upper()]
