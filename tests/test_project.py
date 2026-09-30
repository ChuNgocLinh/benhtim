from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import blood_pressure_follow_up, has_current_warning, supplemental_assessment
from src.data_utils import prepare_dataset
from src.predict import predict_probability
from src.records import load_patient_records, save_patient_record
from src.settings import (
    DATASET_SOURCES_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_INFO,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


class HeartDiseaseProjectTest(unittest.TestCase):
    def test_dataset_is_available_and_binary(self) -> None:
        self.assertTrue(RAW_DATA_PATH.exists(), "Expanded UCI raw dataset is missing")
        self.assertTrue(DATASET_SOURCES_PATH.exists(), "UCI source report is missing")
        data = prepare_dataset()

        self.assertEqual(len(data), 920)
        self.assertTrue(set(data[TARGET_COLUMN].unique()).issubset({0, 1}))
        self.assertEqual(list(data[FEATURES].columns), FEATURES)

    def test_model_metadata_contains_clinical_threshold(self) -> None:
        self.assertTrue(MODEL_PATH.exists(), "Trained model artifact is missing")
        self.assertTrue(METADATA_PATH.exists(), "Model metadata is missing")

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        threshold = float(metadata["decision_threshold"])
        test_metrics = metadata["test_metrics"]

        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        for metric in ["accuracy", "precision", "recall", "roc_auc", "brier_score"]:
            self.assertIn(metric, test_metrics)
            self.assertGreaterEqual(float(test_metrics[metric]), 0.0)

        intervals = metadata.get("test_metric_intervals", {})
        for metric in ["precision", "recall", "specificity", "roc_auc", "brier_score"]:
            self.assertIn(metric, intervals)
            self.assertLessEqual(float(intervals[metric]["lower"]), float(intervals[metric]["upper"]))
            self.assertGreaterEqual(float(intervals[metric]["lower"]), 0.0)
            self.assertLessEqual(float(intervals[metric]["upper"]), 1.0)

        confusion = metadata["confusion_matrix"]
        self.assertEqual(len(confusion), 2)
        self.assertEqual(sum(sum(row) for row in confusion), metadata["split"]["test"])

    def test_model_predicts_valid_probability(self) -> None:
        import joblib

        model = joblib.load(MODEL_PATH)
        sample = {feature: float(FEATURE_INFO[feature]["default"]) for feature in FEATURES}
        probability = predict_probability(model, sample)

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_feature_importance_is_saved(self) -> None:
        self.assertTrue(FEATURE_IMPORTANCE_PATH.exists(), "Feature importance file is missing")
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

        self.assertFalse(importance.empty)
        self.assertIn("feature", importance.columns)
        self.assertIn("importance_mean", importance.columns)

    def test_patient_record_is_saved_and_loaded_with_name_and_result(self) -> None:
        values = {feature: float(FEATURE_INFO[feature]["default"]) for feature in FEATURES}
        values.update(
            {
                "trestbps_diastolic": None,
                "current_symptoms": 0.0,
                "symptom_info_unclear": 0.0,
                "symptom_radiation": 1.0,
                "symptom_duration": 2.0,
                "symptom_severity": 5.0,
                "symptom_dyspnea": 1.0,
                "symptom_cold_sweat": 0.0,
                "symptom_nausea": 0.0,
                "symptom_dizziness_syncope": 0.0,
                "symptom_palpitations": 0.0,
                "current_chest_pain": 0.0,
                "current_severe_dyspnea": 0.0,
                "current_syncope": 0.0,
                "current_cold_sweat": 0.0,
                "current_radiating_pain": 0.0,
                "symptom_onset": 2.0,
                "symptom_at_rest": 0.0,
                "symptom_worsening": 1.0,
                "symptom_orthopnea": 1.0,
                "symptom_nocturnal_dyspnea": 0.0,
                "symptom_leg_swelling": 0.0,
                "history_hypertension": 1.0,
                "history_diabetes": 0.0,
                "history_dyslipidemia": 1.0,
                "history_cardiovascular": 0.0,
                "family_history_early_cvd": 0.0,
                "smoking_status": 2.0,
                "resting_hr": 78.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "patient_records.csv"
            record_id = save_patient_record(
                "  Nguyễn   Văn A ",
                values,
                probability=0.42,
                risk_level="Mức dự đoán trung bình",
                decision_threshold=0.225,
                screening_label="Cần theo dõi thêm",
                path=path,
            )
            records = load_patient_records(path)

        self.assertEqual(len(record_id), 32)
        self.assertEqual(len(records), 1)
        self.assertEqual(records.iloc[0]["patient_name"], "Nguyễn Văn A")
        self.assertAlmostEqual(float(records.iloc[0]["probability"]), 0.42)
        self.assertEqual(records.iloc[0]["risk_level"], "Mức dự đoán trung bình")
        self.assertEqual(float(records.iloc[0]["symptom_radiation"]), 1.0)
        self.assertEqual(float(records.iloc[0]["symptom_dyspnea"]), 1.0)
        self.assertEqual(float(records.iloc[0]["history_hypertension"]), 1.0)
        self.assertEqual(float(records.iloc[0]["resting_hr"]), 78.0)

    def test_current_warning_uses_individual_red_flags(self) -> None:
        self.assertFalse(has_current_warning({"current_symptoms": 0.0}))
        self.assertTrue(has_current_warning({"current_chest_pain": 1.0}))
        self.assertTrue(has_current_warning({"current_radiating_pain": 1.0}))

    def test_supplemental_assessment_stays_separate_from_model_probability(self) -> None:
        urgent = supplemental_assessment({"current_chest_pain": 1.0})
        self.assertEqual(urgent[2], "critical")
        follow_up = supplemental_assessment({"history_hypertension": 1.0, "history_diabetes": 1.0})
        self.assertEqual(follow_up[2], "warning")
        clear = supplemental_assessment({})
        self.assertEqual(clear[2], "info")

    def test_blood_pressure_follow_up_covers_low_high_and_critical(self) -> None:
        low = " ".join(blood_pressure_follow_up(85)).lower()
        high = " ".join(blood_pressure_follow_up(150)).lower()
        critical = " ".join(blood_pressure_follow_up(185)).lower()

        self.assertIn("thấp", low)
        self.assertIn("đo lại", low)
        self.assertIn("ăn nhạt", high)
        self.assertIn("không tự tăng, giảm hoặc ngừng thuốc", high)
        self.assertIn("gọi cấp cứu", critical)
        self.assertIn("liên hệ cơ sở y tế ngay trong ngày", critical)


if __name__ == "__main__":
    unittest.main()
