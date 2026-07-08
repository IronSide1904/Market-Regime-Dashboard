from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from config import HMM_TICKERS


CORE_HMM_TICKERS = {"SPY", "QQQ", "^VIX"}
MIN_HMM_DAYS = 504
MIN_HMM_FEATURES = 5


@dataclass(frozen=True)
class HMMResult:
    available: bool
    regime: str
    confidence: float
    transition_risk: str
    bull_probability: float
    neutral_probability: float
    stress_probability: float
    feature_count: int
    feature_names: list[str]
    last_updated: pd.Timestamp | None
    warnings: list[str]


def build_hmm_result(period: str = "5y") -> HMMResult:
    warnings: list[str] = []

    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _empty_result(
            "HMM dependencies are not installed. Install hmmlearn and scikit-learn to enable the HMM regime model."
        )

    close = _download_hmm_close(period=period)
    if close.empty:
        return _empty_result("HMM market data is unavailable.")

    missing_core = sorted(ticker for ticker in CORE_HMM_TICKERS if ticker not in close.columns or close[ticker].dropna().empty)
    if missing_core:
        return _empty_result(f"HMM core ticker data is missing: {', '.join(missing_core)}.")

    for ticker in HMM_TICKERS:
        if ticker not in close.columns or close[ticker].dropna().shape[0] < MIN_HMM_DAYS:
            if ticker in CORE_HMM_TICKERS:
                return _empty_result(f"HMM core ticker {ticker} does not have enough history.")
            warnings.append(f"HMM optional ticker {ticker} is missing or has limited history; related features were skipped.")

    raw_features, model_features = build_hmm_features(close)
    if model_features.shape[1] < MIN_HMM_FEATURES:
        return _empty_result(
            f"HMM needs at least {MIN_HMM_FEATURES} valid features; only {model_features.shape[1]} are available."
        )
    if model_features.shape[0] < MIN_HMM_DAYS:
        return _empty_result(
            f"HMM needs at least {MIN_HMM_DAYS} clean trading days; only {model_features.shape[0]} are available."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_features)
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
    )

    try:
        states = model.fit(scaled).predict(scaled)
        probabilities = model.predict_proba(scaled)[-1]
    except Exception as exc:
        return _empty_result(f"HMM model fitting failed: {exc}")

    state_labels = _label_states(raw_features=raw_features.loc[model_features.index], states=states)
    label_probabilities = _label_probabilities(probabilities=probabilities, state_labels=state_labels)
    regime = max(label_probabilities, key=label_probabilities.get)
    confidence = label_probabilities[regime]
    stress_probability = label_probabilities["Stress / Risk-Off"]
    transition_risk = _transition_risk(confidence=confidence, stress_probability=stress_probability)

    return HMMResult(
        available=True,
        regime=regime,
        confidence=float(confidence),
        transition_risk=transition_risk,
        bull_probability=float(label_probabilities["Bull / Calm"]),
        neutral_probability=float(label_probabilities["Neutral / Choppy"]),
        stress_probability=float(stress_probability),
        feature_count=int(model_features.shape[1]),
        feature_names=list(model_features.columns),
        last_updated=pd.Timestamp(model_features.index[-1]),
        warnings=warnings,
    )


def build_hmm_features(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: dict[str, pd.Series] = {}

    _add_feature(features, "spy_return", _daily_return(close, "SPY"))
    _add_feature(features, "qqq_return", _daily_return(close, "QQQ"))
    _add_feature(features, "iwm_return", _daily_return(close, "IWM"))
    _add_feature(features, "spy_realized_vol_20d", _realized_vol(close, "SPY", 20))
    _add_feature(features, "vix_relative_50d", _ratio_to_sma(close, "^VIX", 50))
    _add_feature(features, "rsp_spy_change_20d", _ratio_change(close, "RSP", "SPY", 20))
    _add_feature(features, "xlk_xlu_change_20d", _ratio_change(close, "XLK", "XLU", 20))
    _add_feature(features, "hyg_tlt_change_20d", _ratio_change(close, "HYG", "TLT", 20))

    raw_features = pd.DataFrame(features)
    if "^VIX" in close.columns:
        raw_features["vix_level"] = close["^VIX"]

    model_features = raw_features.replace([np.inf, -np.inf], np.nan)
    model_features = model_features.drop(columns=["vix_level"], errors="ignore").dropna()
    raw_features = raw_features.loc[model_features.index]
    return raw_features, model_features


def _download_hmm_close(period: str) -> pd.DataFrame:
    downloaded = yf.download(
        HMM_TICKERS,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        close = downloaded["Close"].copy()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].copy()
        close.columns = [HMM_TICKERS[0]]
    else:
        return pd.DataFrame()

    return close.dropna(how="all")


def _add_feature(features: dict[str, pd.Series], name: str, series: pd.Series | None) -> None:
    if series is not None and not series.dropna().empty:
        features[name] = series


def _daily_return(close: pd.DataFrame, ticker: str) -> pd.Series | None:
    if ticker not in close.columns:
        return None
    return close[ticker].pct_change()


def _realized_vol(close: pd.DataFrame, ticker: str, window: int) -> pd.Series | None:
    if ticker not in close.columns:
        return None
    return close[ticker].pct_change().rolling(window).std() * np.sqrt(252)


def _ratio_to_sma(close: pd.DataFrame, ticker: str, window: int) -> pd.Series | None:
    if ticker not in close.columns:
        return None
    return close[ticker] / close[ticker].rolling(window).mean()


def _ratio_change(close: pd.DataFrame, numerator: str, denominator: str, window: int) -> pd.Series | None:
    if numerator not in close.columns or denominator not in close.columns:
        return None
    ratio = close[numerator] / close[denominator]
    return ratio.pct_change(window)


def _label_states(raw_features: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    state_scores = {}
    for state in sorted(set(states)):
        state_frame = raw_features.loc[states == state]
        means = state_frame.mean(numeric_only=True)
        return_score = (
            means.get("spy_return", 0.0)
            + means.get("qqq_return", 0.0)
            + means.get("iwm_return", 0.0)
        )
        ratio_score = (
            means.get("rsp_spy_change_20d", 0.0)
            + means.get("xlk_xlu_change_20d", 0.0)
            + means.get("hyg_tlt_change_20d", 0.0)
        )
        vol_penalty = means.get("spy_realized_vol_20d", 0.0) + max(means.get("vix_relative_50d", 1.0) - 1.0, 0.0)
        state_scores[state] = float(return_score + ratio_score - vol_penalty)

    ranked_states = sorted(state_scores, key=state_scores.get, reverse=True)
    labels = {
        ranked_states[0]: "Bull / Calm",
        ranked_states[-1]: "Stress / Risk-Off",
    }
    for state in ranked_states[1:-1]:
        labels[state] = "Neutral / Choppy"
    return labels


def _label_probabilities(probabilities: np.ndarray, state_labels: dict[int, str]) -> dict[str, float]:
    label_probabilities = {
        "Bull / Calm": 0.0,
        "Neutral / Choppy": 0.0,
        "Stress / Risk-Off": 0.0,
    }
    for state, probability in enumerate(probabilities):
        label = state_labels.get(state, "Neutral / Choppy")
        label_probabilities[label] += float(probability)
    return label_probabilities


def _transition_risk(confidence: float, stress_probability: float) -> str:
    if stress_probability > 0.50:
        return "High"
    if stress_probability > 0.30:
        return "Elevated"
    if confidence >= 0.75:
        return "Low"
    if confidence >= 0.55:
        return "Medium"
    return "High"


def _empty_result(warning: str) -> HMMResult:
    return HMMResult(
        available=False,
        regime="Unavailable",
        confidence=0.0,
        transition_risk="Unavailable",
        bull_probability=0.0,
        neutral_probability=0.0,
        stress_probability=0.0,
        feature_count=0,
        feature_names=[],
        last_updated=None,
        warnings=[warning],
    )
