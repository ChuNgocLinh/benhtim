from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .settings import FEATURES, PATIENT_RECORDS_PATH


SUPPLEMENTAL_RECORD_COLUMNS = [
    "current_chest_pain",
    "current_severe_dyspnea",
    "current_syncope",
    "current_cold_sweat",
    "current_radiating_pain",
    "symptom_onset",
    "symptom_at_rest",
    "symptom_worsening",
    "symptom_orthopnea",
    "symptom_nocturnal_dyspnea",
    "symptom_leg_swelling",
    "history_hypertension",
    "history_diabetes",
    "history_dyslipidemia",
    "history_cardiovascular",
    "family_history_early_cvd",
    "smoking_status",
    "resting_hr",
]


RECORD_COLUMNS = [
    "record_id",
    "saved_at",
    "patient_name",
    "probability",
    "risk_level",
    "decision_threshold",
    "screening_label",
    *FEATURES,
    "trestbps_diastolic",
    "current_symptoms",
    "symptom_info_unclear",
    "symptom_radiation",
    "symptom_duration",
    "symptom_severity",
    "symptom_dyspnea",
    "symptom_cold_sweat",
    "symptom_nausea",
    "symptom_dizziness_syncope",
    "symptom_palpitations",
    *SUPPLEMENTAL_RECORD_COLUMNS,
]


def _clean_patient_name(patient_name: str) -> str:
    cleaned = " ".join(str(patient_name).split())
    if not cleaned:
        raise ValueError("Tên bệnh nhân không được để trống.")
    return cleaned[:120]


def save_patient_record(
    patient_name: str,
    values: dict[str, float | None],
    probability: float,
    risk_level: str,
    decision_threshold: float,
    screening_label: str,
    path: Path = PATIENT_RECORDS_PATH,
) -> str:
    """Append one local patient assessment and return its stable record id."""
    record_id = uuid4().hex
    record = {
        "record_id": record_id,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "patient_name": _clean_patient_name(patient_name),
        "probability": float(probability),
        "risk_level": risk_level,
        "decision_threshold": float(decision_threshold),
        "screening_label": screening_label,
        **{feature: values[feature] for feature in FEATURES},
        "trestbps_diastolic": values.get("trestbps_diastolic"),
        "current_symptoms": values.get("current_symptoms", 0.0),
        "symptom_info_unclear": values.get("symptom_info_unclear", 0.0),
        "symptom_radiation": values.get("symptom_radiation", 0.0),
        "symptom_duration": values.get("symptom_duration", 0.0),
        "symptom_severity": values.get("symptom_severity", 0.0),
        "symptom_dyspnea": values.get("symptom_dyspnea", 0.0),
        "symptom_cold_sweat": values.get("symptom_cold_sweat", 0.0),
        "symptom_nausea": values.get("symptom_nausea", 0.0),
        "symptom_dizziness_syncope": values.get("symptom_dizziness_syncope", 0.0),
        "symptom_palpitations": values.get("symptom_palpitations", 0.0),
        **{
            column: values.get(column)
            for column in SUPPLEMENTAL_RECORD_COLUMNS
        },
    }
    frame = pd.DataFrame([record], columns=RECORD_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        for column in RECORD_COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
        existing = existing[RECORD_COLUMNS]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return record_id


def load_patient_records(path: Path = PATIENT_RECORDS_PATH) -> pd.DataFrame:
    """Load saved assessments, returning a stable empty shape when none exist."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    frame = pd.read_csv(path)
    for column in RECORD_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[RECORD_COLUMNS].sort_values("saved_at", ascending=False).reset_index(drop=True)
