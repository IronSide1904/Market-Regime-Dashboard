MARKET_TICKERS = {
    "VIX": "^VIX",
    "RSP": "RSP",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "XLK": "XLK",
    "XLU": "XLU",
    "HYG": "HYG",
    "TLT": "TLT",
}

DEFAULT_TICKER = "SPY"
DEFAULT_BENCHMARK = "SPY"
BENCHMARKS = ["SPY", "QQQ"]
HMM_TICKERS = ["SPY", "QQQ", "IWM", "RSP", "^VIX", "XLK", "XLU", "HYG", "TLT"]

TIMEFRAMES = {
    "5D": "3y",
    "10D": "3y",
    "1M": "3y",
    "3M": "3y",
    "QTD": "3y",
    "YTD": "3y",
    "6M": "3y",
    "1Y": "3y",
    "3Y": "3y",
    "5Y": "5y",
}

SWING_TIMEFRAMES = ["5D", "10D", "1M", "3M", "QTD", "YTD", "6M", "1Y"]

SENSITIVITY_LOOKBACKS = {
    "Conservative": {
        "trend": 220,
        "vix": 60,
        "relative_strength": 60,
        "breadth": 60,
        "leadership": 60,
    },
    "Balanced": {
        "trend": 200,
        "vix": 50,
        "relative_strength": 50,
        "breadth": 50,
        "leadership": 50,
    },
    "Aggressive": {
        "trend": 150,
        "vix": 40,
        "relative_strength": 40,
        "breadth": 40,
        "leadership": 40,
    },
}

MIN_TRADING_DAYS = 252

SEARCHABLE_SIGNAL_WEIGHTS = {
    "Ticker Trend": 25,
    "Benchmark Trend": 20,
    "VIX Regime": 15,
    "Relative Strength": 15,
    "Market Breadth / Leadership": 10,
}

VOLUME_CONFIG = {
    "rvol_high": 1.5,
    "rvol_extreme": 2.5,
    "volume_percentile_high": 80,
    "volume_percentile_extreme": 95,
    "weak_volume_rvol": 0.8,
    "sharp_down_day_pct": -2.0,
    "max_positive_adjustment": 15,
    "max_negative_adjustment": -15,
}

MANUAL_FLOAT_SHARES = {
    # "NVDA": 23000000000,
    # "AAPL": 15000000000,
}

THEME_GROUPS = {
    "AI_SEMIS": ["NVDA", "AMD", "AVGO", "MRVL", "MU"],
    "MEGA_CAP_TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "INDEX_ETFS": ["SPY", "QQQ", "IWM", "DIA"],
    "CRYPTO_PROXIES": ["COIN", "MSTR", "IBIT"],
}

LEVERAGED_TICKER_MAP = {
    "QQQ": {
        "bull": ["TQQQ"],
        "bear": ["SQQQ"],
    },
    "SPY": {
        "bull": ["SPXL"],
        "bear": ["SPXS"],
    },
}

FINVIZ_CONFIG = {
    "enabled": True,
    "base_url": "https://elite.finviz.com/export/screener",
    "default_filters": "",
    "view": "152",
    "filter_type": "4",
    "timeout": 20,
    "cache_minutes": 60,
}

FINVIZ_COLUMNS = {
    "ticker": 1,
    "company": 2,
    "sector": 3,
    "industry": 4,
    "market_cap": 6,
    "shares_outstanding": 24,
    "shares_float": 25,
    "short_float": 30,
    "beta": 48,
    "atr": 49,
    "volatility_week": 50,
    "volatility_month": 51,
    "change_from_open": 60,
    "gap": 61,
    "average_volume": 63,
    "relative_volume": 64,
    "price": 65,
    "change": 66,
    "volume": 67,
    "short_interest": 84,
    "float_percent": 85,
    "trades": 89,
    "after_hours_volume": 141,
}

DEBUG_MODE = False

REGIME_RULES = {
    "Risk-On": {
        "min_score": 75,
        "max_score": 100,
        "exposure": 1.0,
        "description": "Favor full or normal equity exposure.",
        "color": "#17803d",
    },
    "Neutral": {
        "min_score": 45,
        "max_score": 74,
        "exposure": 0.6,
        "description": "Reduce size and be selective.",
        "color": "#c99400",
    },
    "Defensive": {
        "min_score": 0,
        "max_score": 44,
        "exposure": 0.2,
        "description": "Hold more cash, hedge, or reduce risk.",
        "color": "#b83232",
    },
}

APP_TITLE = "MR-1 Lite"
