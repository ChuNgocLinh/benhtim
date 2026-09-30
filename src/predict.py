from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .settings import FEATURE_INFO, FEATURES, METADATA_PATH, MODEL_PATH, RISK_LEVELS


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def patient_dataframe(values: dict[str, float]) -> pd.DataFrame:
    normalized = {feature: float(values[feature]) for feature in FEATURES}
    return pd.DataFrame([normalized], columns=FEATURES)


def predict_probability(model, values: dict[str, float]) -> float:
    frame = patient_dataframe(values)
    probability = model.predict_proba(frame)[0, 1]
    return float(probability)


def risk_level(probability: float) -> dict:
    for level in RISK_LEVELS:
        if level["min"] <= probability < level["max"]:
            return level
    return RISK_LEVELS[-1]


def explain_by_reference_delta(
    model,
    values: dict[str, float],
    reference_values: dict[str, float],
) -> pd.DataFrame:
    patient_probability = predict_probability(model, values)
    rows = []
    for feature in FEATURES:
        reference_case = dict(values)
        reference_case[feature] = float(reference_values[feature])
        reference_probability = predict_probability(model, reference_case)
        delta = patient_probability - reference_probability
        info = FEATURE_INFO[feature]
        rows.append(
            {
                "feature": feature,
                "label_vi": info["label_vi"],
                "patient_value": values[feature],
                "reference_value": reference_values[feature],
                "delta_probability": float(delta),
                "direction": "Tăng nguy cơ" if delta > 0 else "Giảm nguy cơ",
                "abs_delta": abs(float(delta)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False)


def display_value(feature: str, value: float) -> str:
    info = FEATURE_INFO[feature]
    options = info.get("options", {})
    key = f"{float(value):.1f}"
    if key in options:
        return options[key]
    unit = info.get("unit", "")
    if unit:
        return f"{value:g} {unit}"
    return f"{value:g}"
