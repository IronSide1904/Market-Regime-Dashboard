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

PEER_OVERRIDE_CONFIG = {
    "enabled": True,
    "default_mode": "Auto",
    "allow_custom_ticker": True,
    "fallback_to_benchmark": True,
    "min_required_rows": 60,
}

PEER_GROUPS = {
    "AMD": ["NVDA", "AVGO", "INTC", "MU", "SMH", "SOXX"],
    "NVDA": ["AMD", "AVGO", "MRVL", "MU", "SMH", "SOXX"],
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "QQQ"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "QQQ"],
    "TSLA": ["RIVN", "LCID", "GM", "F", "XLY"],
    "SPY": ["QQQ", "IWM", "DIA", "RSP"],
    "QQQ": ["SPY", "IWM", "DIA", "XLK"],
}

RELATIVE_CONTEXT_CONFIG = {
    "enabled": True,
    "default_benchmark": "SPY",
    "secondary_benchmark": "QQQ",
    "relative_z_window": 60,
    "correlation_windows": [20, 60, 120],
    "beta_windows": [20, 60],
    "relative_trend_threshold": 0.03,
    "correlation_drop_warning": 0.20,
    "correlation_unstable_level": 0.25,
    "beta_change_threshold": 0.25,
    "strong_extension_z": 2.0,
    "moderate_extension_z": 1.0,
    "max_score_adjustment": 10,
    "min_score_adjustment": -10,
}

RELATIVE_TREND_QUALITY_CONFIG = {
    "enabled": True,
    "weights": {
        "relative_trend_strength": 40,
        "relationship_stability": 30,
        "volume_confirmation": 30,
    },
    "labels": {
        "clean": 80,
        "good_but_watch": 60,
        "mixed": 40,
    },
    "corr_thresholds": {
        "strong": 0.50,
        "moderate": 0.30,
        "max_correlation_change": 0.25,
    },
    "show_in_overview": True,
    "show_in_recommendation": True,
    "show_in_swing_tab": True,
}

COMBINED_RISK_OVERLAY_CONFIG = {
    "enabled": True,
    "weights": {
        "mr1_score": 0.35,
        "clean_relative_trend_score": 0.25,
        "swing_score": 0.20,
        "volume_confirmation": 0.10,
        "swing_volatility": 0.10,
    },
    "labels": {
        "full_risk_allowed": 80,
        "risk_allowed": 65,
        "selective_risk": 50,
        "reduce_risk": 35,
    },
    "base_exposure": {
        "full_risk_allowed": 1.00,
        "risk_allowed": 0.80,
        "selective_risk": 0.50,
        "reduce_risk": 0.30,
        "avoid_defensive": 0.10,
    },
    "position_size_multiplier": {
        "full_risk_allowed": 1.00,
        "risk_allowed": 0.75,
        "selective_risk": 0.50,
        "reduce_risk": 0.25,
        "avoid_defensive": 0.00,
    },
    "risk_caps": {
        "defensive_mr1_max_exposure": 0.30,
        "risk_off_max_exposure": 0.15,
        "panic_volume_max_exposure": 0.20,
        "distribution_volume_max_exposure": 0.40,
        "high_volatility_max_position_multiplier": 0.50,
        "weak_clean_trend_threshold": 40,
    },
}

RELATIVE_TREND_TIMEFRAME_WINDOWS = {
    "5D": {"rs_lookback": 5, "rs_ma": 5, "corr_short": 5, "corr_long": 20},
    "10D": {"rs_lookback": 10, "rs_ma": 10, "corr_short": 10, "corr_long": 30},
    "1M": {"rs_lookback": 21, "rs_ma": 20, "corr_short": 20, "corr_long": 60},
    "2M": {"rs_lookback": 42, "rs_ma": 30, "corr_short": 25, "corr_long": 75},
    "3M": {"rs_lookback": 63, "rs_ma": 50, "corr_short": 30, "corr_long": 90},
    "4M": {"rs_lookback": 84, "rs_ma": 75, "corr_short": 45, "corr_long": 105},
    "6M": {"rs_lookback": 126, "rs_ma": 100, "corr_short": 60, "corr_long": 126},
    "8M": {"rs_lookback": 168, "rs_ma": 150, "corr_short": 75, "corr_long": 168},
    "10M": {"rs_lookback": 210, "rs_ma": 180, "corr_short": 90, "corr_long": 210},
    "YTD": {"rs_lookback": 126, "rs_ma": 100, "corr_short": 60, "corr_long": 126},
    "1Y": {"rs_lookback": 252, "rs_ma": 200, "corr_short": 90, "corr_long": 252},
}

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

COMMON_TIMEFRAME_OPTIONS = ["5D", "10D", "1M", "2M", "3M", "4M", "6M", "8M", "10M", "YTD", "1Y"]

TIMEFRAMES = {label: "3y" for label in COMMON_TIMEFRAME_OPTIONS}

SWING_TIMEFRAMES = COMMON_TIMEFRAME_OPTIONS

SENSITIVITY_LOOKBACKS = {
    "Very Conservative": {
        "trend": 240,
        "vix": 70,
        "relative_strength": 70,
        "breadth": 70,
        "leadership": 70,
    },
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
    "Very Aggressive": {
        "trend": 120,
        "vix": 25,
        "relative_strength": 25,
        "breadth": 25,
        "leadership": 25,
    },
}

TIMEFRAME_SCORE_PROFILES = {
    "5D": {
        "scope": "Tactical",
        "description": "Fastest MR-1 read for short tactical risk shifts.",
        "lookbacks": {"trend": 35, "vix": 10, "relative_strength": 10, "breadth": 10, "leadership": 10},
    },
    "10D": {
        "scope": "Tactical",
        "description": "Fast MR-1 read for near-term tactical positioning.",
        "lookbacks": {"trend": 45, "vix": 14, "relative_strength": 14, "breadth": 14, "leadership": 14},
    },
    "1M": {
        "scope": "Swing",
        "description": "Medium-fast MR-1 read for swing-trading conditions.",
        "lookbacks": {"trend": 60, "vix": 20, "relative_strength": 20, "breadth": 20, "leadership": 20},
    },
    "2M": {
        "scope": "Swing",
        "description": "Two-month swing read with a little more smoothing than 1M.",
        "lookbacks": {"trend": 75, "vix": 25, "relative_strength": 25, "breadth": 25, "leadership": 25},
    },
    "3M": {
        "scope": "Swing",
        "description": "Balanced swing horizon with less noise than the 1M read.",
        "lookbacks": {"trend": 90, "vix": 30, "relative_strength": 30, "breadth": 30, "leadership": 30},
    },
    "4M": {
        "scope": "Position",
        "description": "Early position read bridging swing and slower allocation horizons.",
        "lookbacks": {"trend": 115, "vix": 38, "relative_strength": 38, "breadth": 38, "leadership": 38},
    },
    "QTD": {
        "scope": "Position",
        "description": "Position-style MR-1 read focused on the current quarter.",
        "lookbacks": {"trend": 120, "vix": 40, "relative_strength": 40, "breadth": 40, "leadership": 40},
    },
    "YTD": {
        "scope": "Position",
        "description": "Position-style MR-1 read for year-to-date risk conditions.",
        "lookbacks": {"trend": 150, "vix": 45, "relative_strength": 45, "breadth": 45, "leadership": 45},
    },
    "6M": {
        "scope": "Position",
        "description": "Position-style MR-1 read with medium-slow smoothing.",
        "lookbacks": {"trend": 170, "vix": 50, "relative_strength": 50, "breadth": 50, "leadership": 50},
    },
    "8M": {
        "scope": "Position",
        "description": "Longer position read with slower confirmation windows.",
        "lookbacks": {"trend": 180, "vix": 52, "relative_strength": 52, "breadth": 52, "leadership": 52},
    },
    "10M": {
        "scope": "Position",
        "description": "Slow position read close to annual allocation context.",
        "lookbacks": {"trend": 185, "vix": 54, "relative_strength": 54, "breadth": 54, "leadership": 54},
    },
    "1Y": {
        "scope": "Position",
        "description": "Slow position horizon for broader risk allocation.",
        "lookbacks": {"trend": 190, "vix": 55, "relative_strength": 55, "breadth": 55, "leadership": 55},
    },
    "3Y": {
        "scope": "Strategic",
        "description": "Classic MR-1 style strategic regime read.",
        "lookbacks": {"trend": 200, "vix": 50, "relative_strength": 50, "breadth": 50, "leadership": 50},
    },
    "5Y": {
        "scope": "Strategic",
        "description": "Slowest strategic regime read for long-history context.",
        "lookbacks": {"trend": 220, "vix": 60, "relative_strength": 60, "breadth": 60, "leadership": 60},
    },
}

SENSITIVITY_FACTORS = {
    "Very Conservative": 1.25,
    "Conservative": 1.12,
    "Balanced": 1.0,
    "Aggressive": 0.8,
    "Very Aggressive": 0.65,
}

SWING_SCORE_PROFILES = {
    "5D": {
        "scope": "Tactical",
        "description": "Short tactical swing score emphasizing 5D/10D momentum and near-term risk.",
        "relative_windows": ["5D", "10D"],
        "support_windows": ["5D", "10D"],
        "peer_windows": ["5D", "10D"],
        "trend_checks": [("20D SMA", 12), ("50D SMA", 4)],
        "stack_checks": [("20D SMA", "50D SMA", 4)],
        "atr_clean": 0.035,
        "atr_warning": 0.06,
    },
    "10D": {
        "scope": "Tactical",
        "description": "Fast tactical swing score using 10D/1M confirmation.",
        "relative_windows": ["10D", "1M"],
        "support_windows": ["10D", "1M"],
        "peer_windows": ["10D", "1M"],
        "trend_checks": [("20D SMA", 10), ("50D SMA", 6)],
        "stack_checks": [("20D SMA", "50D SMA", 4)],
        "atr_clean": 0.04,
        "atr_warning": 0.065,
    },
    "1M": {
        "scope": "Swing",
        "description": "Standard swing score using 1M/3M relative strength.",
        "relative_windows": ["1M", "3M"],
        "support_windows": ["1M", "3M", "QTD"],
        "peer_windows": ["1M", "3M", "QTD"],
        "trend_checks": [("20D SMA", 8), ("50D SMA", 8)],
        "stack_checks": [("20D SMA", "50D SMA", 4)],
        "atr_clean": 0.04,
        "atr_warning": 0.07,
    },
    "2M": {
        "scope": "Swing",
        "description": "Two-month swing score using 1M/2M/3M confirmation.",
        "relative_windows": ["1M", "2M", "3M"],
        "support_windows": ["1M", "2M", "3M"],
        "peer_windows": ["1M", "2M", "3M"],
        "trend_checks": [("20D SMA", 6), ("50D SMA", 10)],
        "stack_checks": [("20D SMA", "50D SMA", 4)],
        "atr_clean": 0.043,
        "atr_warning": 0.073,
    },
    "3M": {
        "scope": "Swing",
        "description": "Normal swing score with more 3M/QTD confirmation.",
        "relative_windows": ["1M", "3M", "QTD"],
        "support_windows": ["1M", "3M", "QTD"],
        "peer_windows": ["1M", "3M", "QTD"],
        "trend_checks": [("50D SMA", 10), ("200D SMA", 6)],
        "stack_checks": [("20D SMA", "50D SMA", 4)],
        "atr_clean": 0.045,
        "atr_warning": 0.075,
    },
    "4M": {
        "scope": "Position",
        "description": "Four-month score emphasizing 3M/4M leadership and trend support.",
        "relative_windows": ["3M", "4M", "6M"],
        "support_windows": ["3M", "4M", "6M"],
        "peer_windows": ["3M", "4M", "6M"],
        "trend_checks": [("50D SMA", 10), ("200D SMA", 6)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.05,
        "atr_warning": 0.08,
    },
    "QTD": {
        "scope": "Position",
        "description": "Current-quarter position score emphasizing QTD and 3M leadership.",
        "relative_windows": ["3M", "QTD"],
        "support_windows": ["3M", "QTD", "6M"],
        "peer_windows": ["3M", "QTD", "6M"],
        "trend_checks": [("50D SMA", 8), ("200D SMA", 8)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.05,
        "atr_warning": 0.08,
    },
    "YTD": {
        "scope": "Position",
        "description": "Position score focused on YTD and 6M trend quality.",
        "relative_windows": ["YTD", "6M"],
        "support_windows": ["YTD", "6M", "1Y"],
        "peer_windows": ["YTD", "6M", "1Y"],
        "trend_checks": [("50D SMA", 6), ("200D SMA", 10)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.055,
        "atr_warning": 0.085,
    },
    "6M": {
        "scope": "Position",
        "description": "Position score emphasizing 6M and 1Y relative strength.",
        "relative_windows": ["6M", "1Y"],
        "support_windows": ["3M", "6M", "1Y"],
        "peer_windows": ["3M", "6M", "1Y"],
        "trend_checks": [("50D SMA", 6), ("200D SMA", 10)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.055,
        "atr_warning": 0.085,
    },
    "8M": {
        "scope": "Position",
        "description": "Eight-month position score emphasizing 6M/8M/1Y trend quality.",
        "relative_windows": ["6M", "8M", "1Y"],
        "support_windows": ["6M", "8M", "1Y"],
        "peer_windows": ["6M", "8M", "1Y"],
        "trend_checks": [("50D SMA", 4), ("200D SMA", 12)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.058,
        "atr_warning": 0.088,
    },
    "10M": {
        "scope": "Position",
        "description": "Ten-month position score focused on broad trend and relative leadership.",
        "relative_windows": ["8M", "10M", "1Y"],
        "support_windows": ["8M", "10M", "1Y"],
        "peer_windows": ["8M", "10M", "1Y"],
        "trend_checks": [("200D SMA", 14)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.06,
        "atr_warning": 0.09,
    },
    "1Y": {
        "scope": "Position",
        "description": "Slow swing/position score with broad trend and 1Y leadership emphasis.",
        "relative_windows": ["6M", "1Y"],
        "support_windows": ["6M", "1Y", "YTD"],
        "peer_windows": ["6M", "1Y", "YTD"],
        "trend_checks": [("200D SMA", 16)],
        "stack_checks": [("50D SMA", "200D SMA", 4)],
        "atr_clean": 0.06,
        "atr_warning": 0.09,
    },
}


def get_timeframe_score_profile(timeframe_label: str, sensitivity: str) -> dict:
    profile = TIMEFRAME_SCORE_PROFILES.get(timeframe_label, TIMEFRAME_SCORE_PROFILES["3Y"])
    factor = SENSITIVITY_FACTORS.get(sensitivity, SENSITIVITY_FACTORS["Balanced"])
    lookbacks = {
        key: max(5, int(round(value * factor)))
        for key, value in profile["lookbacks"].items()
    }
    return {
        "timeframe": timeframe_label,
        "sensitivity": sensitivity,
        "scope": profile["scope"],
        "description": profile["description"],
        "lookbacks": lookbacks,
    }


def get_swing_timeframe_profile(swing_timeframe: str) -> dict:
    profile = SWING_SCORE_PROFILES.get(swing_timeframe, SWING_SCORE_PROFILES["1M"])
    return {
        "timeframe": swing_timeframe,
        **profile,
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

TIMEFRAME_PRESETS = {
    "5D": {
        "description": "Fast tactical setup using short volume and volatility windows.",
        "volume_short_window": 5,
        "volume_medium_window": 10,
        "volume_long_window": 20,
        "volume_percentile_window": 63,
        "float_turnover_windows": [1, 5, 10],
        "atr_window": 10,
        "realized_vol_window": 10,
        "trend_window": 20,
    },
    "10D": {
        "description": "Fast tactical setup using short volume and volatility windows.",
        "volume_short_window": 5,
        "volume_medium_window": 10,
        "volume_long_window": 20,
        "volume_percentile_window": 63,
        "float_turnover_windows": [1, 5, 10],
        "atr_window": 10,
        "realized_vol_window": 10,
        "trend_window": 20,
    },
    "1M": {
        "description": "Standard swing setup using medium volume and volatility windows.",
        "volume_short_window": 10,
        "volume_medium_window": 20,
        "volume_long_window": 50,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 5, 20],
        "atr_window": 14,
        "realized_vol_window": 20,
        "trend_window": 50,
    },
    "2M": {
        "description": "Two-month swing setup using medium volume and volatility windows.",
        "volume_short_window": 10,
        "volume_medium_window": 20,
        "volume_long_window": 42,
        "volume_percentile_window": 126,
        "float_turnover_windows": [1, 10, 20],
        "atr_window": 14,
        "realized_vol_window": 20,
        "trend_window": 50,
    },
    "3M": {
        "description": "Standard swing setup using medium volume and volatility windows.",
        "volume_short_window": 10,
        "volume_medium_window": 20,
        "volume_long_window": 50,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 5, 20],
        "atr_window": 14,
        "realized_vol_window": 20,
        "trend_window": 50,
    },
    "4M": {
        "description": "Four-month setup bridging swing and position windows.",
        "volume_short_window": 14,
        "volume_medium_window": 30,
        "volume_long_window": 84,
        "volume_percentile_window": 168,
        "float_turnover_windows": [1, 20, 40],
        "atr_window": 18,
        "realized_vol_window": 25,
        "trend_window": 75,
    },
    "6M": {
        "description": "Position-style setup using slower volume and volatility windows.",
        "volume_short_window": 20,
        "volume_medium_window": 50,
        "volume_long_window": 63,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 20, 50],
        "atr_window": 21,
        "realized_vol_window": 30,
        "trend_window": 100,
    },
    "8M": {
        "description": "Eight-month position setup using broad confirmation windows.",
        "volume_short_window": 20,
        "volume_medium_window": 63,
        "volume_long_window": 84,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 20, 63],
        "atr_window": 21,
        "realized_vol_window": 42,
        "trend_window": 120,
    },
    "10M": {
        "description": "Ten-month position setup using slow confirmation windows.",
        "volume_short_window": 21,
        "volume_medium_window": 63,
        "volume_long_window": 105,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 21, 63],
        "atr_window": 21,
        "realized_vol_window": 50,
        "trend_window": 150,
    },
    "YTD": {
        "description": "Year-to-date setup using position-style volume and volatility windows.",
        "volume_short_window": 20,
        "volume_medium_window": 50,
        "volume_long_window": 63,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 20, 50],
        "atr_window": 21,
        "realized_vol_window": 30,
        "trend_window": 100,
    },
    "1Y": {
        "description": "Position-style setup using slower volume and volatility windows.",
        "volume_short_window": 20,
        "volume_medium_window": 50,
        "volume_long_window": 63,
        "volume_percentile_window": 252,
        "float_turnover_windows": [1, 20, 50],
        "atr_window": 21,
        "realized_vol_window": 30,
        "trend_window": 100,
    },
}

DEFAULT_TIMEFRAME_PRESET = "1M"

TIMEFRAME_TO_VOLUME_PRESET = {label: label for label in COMMON_TIMEFRAME_OPTIONS}

SWING_TIMEFRAME_TO_VOLATILITY_PRESET = {label: label for label in COMMON_TIMEFRAME_OPTIONS}

CUSTOM_TIMEFRAME_LIMITS = {
    "min_volume_window": 5,
    "max_volume_window": 252,
    "min_atr_window": 5,
    "max_atr_window": 63,
    "min_realized_vol_window": 5,
    "max_realized_vol_window": 126,
}

VOLUME_CONTEXT_CONFIG = {
    "enabled": True,
    "use_timeframe_presets": True,
    "default_preset": DEFAULT_TIMEFRAME_PRESET,
    "show_advanced_controls": False,
}

SWING_VOLATILITY_CONFIG = {
    "enabled": True,
    "use_timeframe_presets": True,
    "default_preset": DEFAULT_TIMEFRAME_PRESET,
    "annualize_realized_vol": True,
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

SCREENER_CONFIG = {
    "enabled": True,
    "max_tickers": 150,
    "default_top_n": 25,
    "default_min_price": 5,
    "default_min_dollar_volume": 10_000_000,
    "cache_ttl_seconds": 3600,
    "default_benchmark": "QQQ",
    "default_market_benchmark": "SPY",
    "default_timeframe": "1M",
    "score_weights": {
        "price_trend": 25,
        "momentum": 25,
        "relative_strength": 30,
        "volume_confirmation": 10,
        "risk_volatility": 10,
    },
}

SCREENER_WATCHLISTS = {
    "Mega Cap Tech": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
    "AI Semis": ["NVDA", "AMD", "AVGO", "MRVL", "MU", "ARM", "TSM", "ASML"],
    "Cybersecurity": ["CRWD", "PANW", "ZS", "S", "FTNT", "OKTA"],
    "Software Growth": ["DDOG", "SNOW", "NET", "MDB", "PLTR", "HUBS"],
    "Crypto Proxies": ["COIN", "MSTR", "MARA", "RIOT", "IBIT"],
}

SCREENER_THEME_GROUPS = {
    "Mega Cap Tech": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
    "AI Semiconductors": ["NVDA", "AMD", "AVGO", "MRVL", "MU", "ARM", "TSM", "ASML"],
    "AI Software": ["PLTR", "SNOW", "DDOG", "NET", "MDB", "AI"],
    "Cybersecurity": ["CRWD", "PANW", "ZS", "FTNT", "OKTA", "S"],
    "Bitcoin Proxies": ["COIN", "MSTR", "MARA", "RIOT", "IBIT"],
    "Ad Tech": ["APP", "TTD", "MGNI", "PUBM", "IAS"],
}

SCREENER_TARGET_TICKER_CONFIG = {
    "enabled": True,
    "default_mode": "Ticker Comparison",
    "include_target_in_results": True,
    "include_benchmarks_in_results": True,
    "highlight_target_row": True,
    "max_auto_peers": 25,
}

SCREENER_PEER_OVERRIDE_CONFIG = {
    "enabled": True,
    "default_enabled": False,
    "default_mode": "append",
    "allowed_modes": ["append", "replace"],
    "default_apply_to": ["Direct Peers", "Sector Peers", "Industry Peers"],
    "always_include_target": True,
    "always_include_benchmarks": True,
    "max_override_tickers": 50,
}

SCREENER_PEER_MAP = {
    "AMD": {
        "theme": "AI Semiconductors",
        "direct_peers": ["NVDA", "INTC", "AVGO", "MRVL", "MU", "ARM", "QCOM", "TSM", "ASML"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
    "NVDA": {
        "theme": "AI Semiconductors",
        "direct_peers": ["AMD", "AVGO", "MRVL", "MU", "ARM", "TSM", "ASML", "INTC"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
    "AAPL": {
        "theme": "Mega Cap Tech",
        "direct_peers": ["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
    "MSFT": {
        "theme": "Mega Cap Tech",
        "direct_peers": ["AAPL", "GOOGL", "AMZN", "META", "NVDA"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
    "META": {
        "theme": "Mega Cap Tech",
        "direct_peers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
    "COIN": {
        "theme": "Bitcoin Proxies",
        "direct_peers": ["MSTR", "MARA", "RIOT", "IBIT"],
        "benchmark": "QQQ",
        "market_benchmark": "SPY",
    },
}

SCREENER_BUCKET_ANALYSIS_CONFIG = {
    "enabled": True,
    "show_bucket_tables": True,
    "show_bucket_charts": True,
    "default_bucket_chart": "Combined Overlay Score",
    "show_overlay_history_chart": True,
    "overlay_history_max_tickers": 10,
    "highlight_target_ticker": True,
    "show_threshold_bands": True,
}

SCREENER_BUCKET_CHART_OPTIONS = [
    "Combined Overlay Score",
    "Momentum Trend Score",
    "Average Daily Float Turnover",
    "RS vs Target",
    "RS vs Benchmark",
    "RS vs Market",
    "RS vs Theme",
    "Timeframe Return",
    "ATR % / Risk",
]

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
    "no": 0,
    "ticker": 1,
    "company": 2,
    "sector": 3,
    "industry": 4,
    "country": 5,
    "market_cap": 6,
    "pe": 7,
    "forward_pe": 8,
    "peg": 9,
    "ps": 10,
    "pb": 11,
    "pc": 12,
    "pfcf": 13,
    "shares_outstanding": 24,
    "shares_float": 25,
    "short_float": 30,
    "short_ratio": 31,
    "roa": 32,
    "roe": 33,
    "roi": 34,
    "current_ratio": 35,
    "quick_ratio": 36,
    "lt_debt_to_equity": 37,
    "debt_to_equity": 38,
    "gross_margin": 39,
    "operating_margin": 40,
    "profit_margin": 41,
    "beta": 48,
    "atr": 49,
    "volatility_week": 50,
    "volatility_month": 51,
    "sma20": 52,
    "sma50": 53,
    "sma200": 54,
    "high52w": 57,
    "low52w": 58,
    "rsi": 59,
    "change_from_open": 60,
    "gap": 61,
    "average_volume": 63,
    "relative_volume": 64,
    "price": 65,
    "change": 66,
    "volume": 67,
    "earnings_date": 68,
    "short_interest": 84,
    "float_percent": 85,
    "trades": 89,
    "after_hours_volume": 141,
}

FINVIZ_DISCOVERY_COLUMNS = ",".join(str(column_id) for column_id in range(0, 111))

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
