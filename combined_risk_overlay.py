from __future__ import annotations

import math

from config import COMBINED_RISK_OVERLAY_CONFIG


def normalize_volume_context_to_score(volume_context: dict | None) -> float | None:
    if not volume_context:
        return None
    context = _text(_get(volume_context, "context", "volume_context", "Volume Context"))
    adjustment = _number(_get(volume_context, "adjustment", "volume_adjustment", "Volume Adjustment"))
    rvol = _number(_get(volume_context, "rvol_medium", "rvol_20d", "RVOL Medium", "RVOL 20D"))
    percentile = _number(_get(volume_context, "volume_percentile", "volume_percentile_1y", "Volume Percentile", "Volume Percentile 1Y"))

    if context in {"Accumulation", "Breakout Confirmation"}:
        score = 90
    elif context == "Neutral":
        score = 55
    elif context == "Weak Participation":
        score = 40
    elif context == "Distribution":
        score = 20
    elif context == "Panic / Liquidation":
        score = 5
    elif context == "Unavailable":
        return None
    else:
        score = 50

    if adjustment is not None:
        score += max(-15, min(15, adjustment * 2))
    if rvol is not None and rvol >= 1.25 and context not in {"Distribution", "Panic / Liquidation"}:
        score += 5
    if percentile is not None and percentile >= 80 and context not in {"Distribution", "Panic / Liquidation"}:
        score += 5
    return _clip(score)


def normalize_swing_volatility_to_score(swing_volatility_context: dict | None) -> float | None:
    if not swing_volatility_context or not swing_volatility_context.get("available", False):
        return None
    status = _text(swing_volatility_context.get("volatility_status"))
    if status == "Low":
        return 85
    if status == "Normal":
        return 75
    if status in {"Elevated", "High"}:
        return 45
    if status == "Extreme":
        return 15
    return 55


def calculate_base_overlay_score(
    mr1_score: float,
    clean_relative_trend_score: float | None,
    swing_score: float | None,
    volume_context: dict | None,
    swing_volatility_context: dict | None,
    config: dict | None = None,
) -> dict:
    settings = config or COMBINED_RISK_OVERLAY_CONFIG
    weights = settings["weights"]
    components = {
        "mr1_score": _number(mr1_score),
        "clean_relative_trend_score": _number(clean_relative_trend_score),
        "swing_score": _number(swing_score),
        "volume_confirmation": normalize_volume_context_to_score(volume_context),
        "swing_volatility": normalize_swing_volatility_to_score(swing_volatility_context),
    }
    missing = [name for name, value in components.items() if value is None]
    if components["mr1_score"] is None:
        return {
            "available": False,
            "overlay_score": 0,
            "component_scores": components,
            "component_contributions": {},
            "missing_components": missing,
            "warnings": ["MR-1 score is unavailable, so Combined Risk Overlay cannot be calculated."],
        }

    contribution = {}
    weighted_sum = 0.0
    used_weight = 0.0
    for name, weight in weights.items():
        value = components.get(name)
        if value is None:
            continue
        contribution[name] = value * weight
        weighted_sum += contribution[name]
        used_weight += weight

    score = weighted_sum / used_weight if used_weight else 0
    return {
        "available": True,
        "overlay_score": int(round(_clip(score))),
        "component_scores": components,
        "component_contributions": contribution,
        "missing_components": missing,
        "warnings": [],
    }


def apply_risk_caps(
    base_overlay: dict,
    mr1_regime: str,
    volume_context: dict | None,
    swing_volatility_context: dict | None,
    clean_relative_trend_score: float | None,
    benchmark_trend_status: str | None,
    config: dict | None = None,
) -> dict:
    settings = config or COMBINED_RISK_OVERLAY_CONFIG
    caps = settings["risk_caps"]
    score = int(base_overlay.get("overlay_score", 0))
    label_key, label = _label_for_score(score, settings)
    recommendation = _recommendation_for_label(label_key)
    exposure = float(settings["base_exposure"][label_key])
    multiplier = float(settings["position_size_multiplier"][label_key])
    risk_cap_applied = False
    cap_reasons: list[str] = []
    warnings: list[str] = list(base_overlay.get("warnings", []))
    allow_buy = True

    regime = _text(mr1_regime)
    volume = _text(_get(volume_context or {}, "context", "volume_context", "Volume Context"))
    volatility = _text((swing_volatility_context or {}).get("volatility_status"))
    clean_score = _number(clean_relative_trend_score)
    benchmark_status = _text(benchmark_trend_status)

    if regime == "Defensive":
        exposure, recommendation = _cap_exposure(exposure, caps["defensive_mr1_max_exposure"], "REDUCE / AVOID")
        risk_cap_applied = True
        cap_reasons.append("MR-1 regime is Defensive")
        warnings.append("Defensive MR-1 regime caps exposure.")
        allow_buy = False
    if regime in {"Risk-Off", "Crisis"}:
        exposure, recommendation = _cap_exposure(exposure, caps["risk_off_max_exposure"], "AVOID")
        risk_cap_applied = True
        cap_reasons.append("Risk-off or crisis regime")
        warnings.append("Risk-off conditions override otherwise positive signals.")
        allow_buy = False
    if volume == "Panic / Liquidation":
        exposure, recommendation = _cap_exposure(exposure, caps["panic_volume_max_exposure"], "AVOID / REDUCE")
        risk_cap_applied = True
        cap_reasons.append("Panic volume is active")
        warnings.append("Panic / Liquidation volume caps risk.")
        allow_buy = False
    elif volume == "Distribution":
        exposure, recommendation = _cap_exposure(exposure, caps["distribution_volume_max_exposure"], "REDUCE")
        risk_cap_applied = True
        cap_reasons.append("Distribution volume is active")
        warnings.append("Distribution volume reduces confidence in the setup.")
        allow_buy = False
    if volatility in {"High", "Elevated", "Extreme"}:
        multiplier = min(multiplier, caps["high_volatility_max_position_multiplier"])
        risk_cap_applied = True
        cap_reasons.append("Swing volatility is elevated" if volatility != "Extreme" else "Swing volatility is extreme")
        warnings.append("Swing volatility limits position size.")
    if clean_score is not None and clean_score < caps["weak_clean_trend_threshold"]:
        recommendation = _more_conservative_recommendation(recommendation, "HOLD / AVOID")
        risk_cap_applied = True
        cap_reasons.append("Clean Relative Trend is weak")
        warnings.append("Weak Clean Relative Trend prevents a clean buy signal.")
        allow_buy = False
    if benchmark_status == "Bearish":
        recommendation = _more_conservative_recommendation(recommendation, "HOLD / SELECTIVE ONLY")
        risk_cap_applied = True
        cap_reasons.append("Benchmark trend is bearish")
        warnings.append("Weak benchmark trend limits new risk.")

    if not allow_buy and recommendation.startswith("BUY"):
        recommendation = "HOLD / AVOID"

    return {
        "overlay_score": score,
        "overlay_label": label,
        "label_key": label_key,
        "final_recommendation": recommendation,
        "suggested_exposure": float(max(0, min(1, exposure))),
        "position_size_multiplier": float(max(0, min(1, multiplier))),
        "risk_cap_applied": risk_cap_applied,
        "risk_cap_reason": "; ".join(_dedupe(cap_reasons)) if cap_reasons else "None",
        "warnings": _dedupe(warnings),
    }


def calculate_combined_risk_overlay(
    mr1_score: float,
    mr1_regime: str,
    clean_relative_trend_score: float | None,
    swing_score: float | None,
    volume_context: dict | None,
    swing_volatility_context: dict | None,
    benchmark_trend_status: str | None,
    data_quality: dict | None = None,
    config: dict | None = None,
) -> dict:
    settings = config or COMBINED_RISK_OVERLAY_CONFIG
    if not settings.get("enabled", True):
        return _unavailable("Combined Risk Overlay is disabled.")

    base = calculate_base_overlay_score(
        mr1_score=mr1_score,
        clean_relative_trend_score=clean_relative_trend_score,
        swing_score=swing_score,
        volume_context=volume_context,
        swing_volatility_context=swing_volatility_context,
        config=settings,
    )
    if not base.get("available", False):
        result = _unavailable("Combined Risk Overlay cannot be calculated without MR-1 score.")
        result["warnings"] = base.get("warnings", result["warnings"])
        return result

    capped = apply_risk_caps(
        base_overlay=base,
        mr1_regime=mr1_regime,
        volume_context=volume_context,
        swing_volatility_context=swing_volatility_context,
        clean_relative_trend_score=clean_relative_trend_score,
        benchmark_trend_status=benchmark_trend_status,
        config=settings,
    )
    positives, warnings = _drivers_and_warnings(
        mr1_score=mr1_score,
        mr1_regime=mr1_regime,
        clean_relative_trend_score=clean_relative_trend_score,
        swing_score=swing_score,
        volume_context=volume_context,
        swing_volatility_context=swing_volatility_context,
        benchmark_trend_status=benchmark_trend_status,
    )
    warnings = _dedupe([*warnings, *capped["warnings"]])
    missing = list(base.get("missing_components", []))
    data_warnings = list((data_quality or {}).get("warnings", []))
    if missing or data_warnings:
        warnings.append("Some data is unavailable, so confidence is reduced.")

    confidence = _confidence(capped["overlay_score"], missing=missing, warnings=warnings, risk_cap=capped["risk_cap_applied"])
    main_support = positives[0] if positives else "No major support is confirmed."
    main_warning = warnings[0] if warnings else "No major warning is active."
    explanation = _explanation(capped, main_support, main_warning)

    return {
        "available": True,
        "base_overlay_score": base["overlay_score"],
        "overlay_score": capped["overlay_score"],
        "overlay_label": capped["overlay_label"],
        "final_recommendation": capped["final_recommendation"],
        "suggested_exposure": capped["suggested_exposure"],
        "position_size_multiplier": capped["position_size_multiplier"],
        "risk_cap_applied": capped["risk_cap_applied"],
        "risk_cap_reason": capped["risk_cap_reason"],
        "confidence": confidence,
        "main_support": main_support,
        "main_warning": main_warning,
        "positive_drivers": _dedupe(positives),
        "warnings": _dedupe(warnings),
        "explanation": explanation,
        "component_scores": base["component_scores"],
        "component_contributions": base["component_contributions"],
        "missing_components": missing,
    }


def _label_for_score(score: int, config: dict) -> tuple[str, str]:
    labels = config["labels"]
    if score >= labels["full_risk_allowed"]:
        return "full_risk_allowed", "Full Risk Allowed"
    if score >= labels["risk_allowed"]:
        return "risk_allowed", "Risk Allowed"
    if score >= labels["selective_risk"]:
        return "selective_risk", "Selective Risk"
    if score >= labels["reduce_risk"]:
        return "reduce_risk", "Reduce Risk"
    return "avoid_defensive", "Avoid / Defensive"


def _recommendation_for_label(label_key: str) -> str:
    return {
        "full_risk_allowed": "BUY / FULL RISK ALLOWED",
        "risk_allowed": "SELECTIVE BUY / RISK ALLOWED",
        "selective_risk": "HOLD / SELECTIVE RISK",
        "reduce_risk": "REDUCE / WAIT",
        "avoid_defensive": "AVOID / DEFENSIVE",
    }[label_key]


def _drivers_and_warnings(
    mr1_score,
    mr1_regime,
    clean_relative_trend_score,
    swing_score,
    volume_context,
    swing_volatility_context,
    benchmark_trend_status,
) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    warnings: list[str] = []
    if _text(mr1_regime) == "Risk-On":
        positives.append("MR-1 regime supports risk-taking")
    elif _text(mr1_regime) == "Defensive":
        warnings.append("MR-1 regime is Defensive")
    if _number(clean_relative_trend_score) is not None:
        if clean_relative_trend_score >= 80:
            positives.append("Clean Relative Trend confirms a high-quality setup")
        elif clean_relative_trend_score < 40:
            warnings.append("Clean Relative Trend is weak")
    else:
        warnings.append("Clean Relative Trend is unavailable")
    if _number(swing_score) is not None:
        if swing_score >= 70:
            positives.append("Swing Score supports tactical timing")
        elif swing_score < 45:
            warnings.append("Swing Score is weak")
    else:
        warnings.append("Swing Score is unavailable")
    volume = _text(_get(volume_context or {}, "context", "volume_context", "Volume Context"))
    if volume in {"Accumulation", "Breakout Confirmation"}:
        positives.append("Volume confirms participation")
    elif volume in {"Weak Participation", "Distribution", "Panic / Liquidation"}:
        warnings.append(f"Volume context is {volume}")
    elif volume == "Neutral":
        warnings.append("Volume confirmation is neutral")
    else:
        warnings.append("Volume context is unavailable")
    volatility = _text((swing_volatility_context or {}).get("volatility_status"))
    if volatility in {"Low", "Normal"}:
        positives.append("Swing volatility allows normal sizing")
    elif volatility in {"Elevated", "High", "Extreme"}:
        warnings.append(f"Swing volatility is {volatility}")
    else:
        warnings.append("Swing volatility is unavailable")
    if _text(benchmark_trend_status) == "Bearish":
        warnings.append("Benchmark trend is bearish")
    elif _text(benchmark_trend_status) == "Bullish":
        positives.append("Benchmark trend supports risk")
    return positives, warnings


def _confidence(score: int, missing: list[str], warnings: list[str], risk_cap: bool) -> str:
    confidence = "High" if score >= 75 and len(warnings) <= 1 else "Medium" if score >= 50 else "Low"
    downgrades = len(missing) + (1 if risk_cap else 0) + (1 if len(warnings) >= 3 else 0)
    for _ in range(downgrades):
        confidence = {"High": "Medium", "Medium": "Low", "Low": "Low"}[confidence]
    return confidence


def _explanation(result: dict, main_support: str, main_warning: str) -> str:
    if result["risk_cap_applied"]:
        return (
            f"{result['overlay_label']}: {main_support}. Risk is capped because {result['risk_cap_reason']}. "
            f"Main warning: {main_warning}."
        )
    return (
        f"{result['overlay_label']}: {main_support}. Suggested exposure and position size follow the combined risk read. "
        f"Main warning: {main_warning}."
    )


def _cap_exposure(current: float, cap: float, recommendation: str) -> tuple[float, str]:
    return min(current, cap), recommendation


def _more_conservative_recommendation(current: str, candidate: str) -> str:
    order = {
        "BUY / FULL RISK ALLOWED": 5,
        "SELECTIVE BUY / RISK ALLOWED": 4,
        "HOLD / SELECTIVE RISK": 3,
        "HOLD / SELECTIVE ONLY": 3,
        "HOLD / AVOID": 2,
        "REDUCE / WAIT": 2,
        "REDUCE": 2,
        "REDUCE / AVOID": 1,
        "AVOID / REDUCE": 1,
        "AVOID / DEFENSIVE": 0,
        "AVOID": 0,
    }
    return candidate if order.get(candidate, 3) < order.get(current, 3) else current


def _unavailable(message: str) -> dict:
    return {
        "available": False,
        "base_overlay_score": 0,
        "overlay_score": 0,
        "overlay_label": "Unavailable",
        "final_recommendation": "HOLD",
        "suggested_exposure": 0.0,
        "position_size_multiplier": 0.0,
        "risk_cap_applied": False,
        "risk_cap_reason": "None",
        "confidence": "Low",
        "main_support": "Unavailable",
        "main_warning": message,
        "positive_drivers": [],
        "warnings": [message],
        "explanation": message,
        "component_scores": {},
        "component_contributions": {},
        "missing_components": [],
    }


def _get(values: dict, *keys):
    for key in keys:
        if key in values:
            return values.get(key)
    return None


def _text(value) -> str:
    return str(value or "Unavailable")


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clip(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        if item and item not in output:
            output.append(item)
    return output
