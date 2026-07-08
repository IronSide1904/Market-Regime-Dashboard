from __future__ import annotations

import pandas as pd


def calculate_indicators(close: pd.DataFrame, lookbacks: dict[str, int]) -> pd.DataFrame:
    data = pd.DataFrame(index=close.index)

    trend_lookback = lookbacks["trend"]
    vix_lookback = lookbacks["vix"]
    rs_lookback = lookbacks["relative_strength"]
    breadth_lookback = lookbacks["breadth"]
    leadership_lookback = lookbacks["leadership"]

    data["Ticker Close"] = close["Ticker"]
    data["Benchmark Close"] = close["Benchmark"]
    data["VIX Close"] = close["VIX"]

    data["Ticker SMA"] = close["Ticker"].rolling(trend_lookback).mean()
    data["Benchmark SMA"] = close["Benchmark"].rolling(trend_lookback).mean()
    data["VIX SMA"] = close["VIX"].rolling(vix_lookback).mean()

    data["Relative Strength"] = close["Ticker"] / close["Benchmark"]
    data["Relative Strength SMA"] = data["Relative Strength"].rolling(rs_lookback).mean()

    data["RSP/SPY"] = close["RSP"] / close["SPY"]
    data["RSP/SPY SMA"] = data["RSP/SPY"].rolling(breadth_lookback).mean()

    data["XLK/XLU"] = close["XLK"] / close["XLU"]
    data["XLK/XLU SMA"] = data["XLK/XLU"].rolling(leadership_lookback).mean()

    return data.dropna()
