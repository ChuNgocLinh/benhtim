from __future__ import annotations

import itertools
import unittest
from math import isfinite

import joblib
import pandas as pd

from src.predict import predict_probability
from src.settings import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, FEATURES, FEATURE_INFO, MODEL_PATH


SUPPLEMENTAL_FIELDS = {
    "current_chest_pain": 1.0,
    "current_severe_dyspnea": 1.0,
    "current_syncope": 1.0,
    "current_cold_sweat": 1.0,
    "current_radiating_pain": 1.0,
    "symptom_dyspnea": 1.0,
    "symptom_cold_sweat": 1.0,
    "symptom_nausea": 1.0,
    "symptom_dizziness_syncope": 1.0,
    "symptom_palpitations": 1.0,
    "symptom_orthopnea": 1.0,
    "symptom_nocturnal_dyspnea": 1.0,
    "symptom_leg_swelling": 1.0,
    "history_hypertension": 1.0,
    "history_diabetes": 1.0,
    "history_dyslipidemia": 1.0,
    "history_cardiovascular": 1.0,
    "family_history_early_cvd": 1.0,
    "smoking_status": 2.0,
    "resting_hr": 78.0,
}


class FeatureOptionValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = joblib.load(MODEL_PATH)
        cls.defaults = {
            feature: float(FEATURE_INFO[feature]["default"])
            for feature in FEATURES
        }

    def assert_valid_probability(self, probability: float) -> None:
        self.assertTrue(isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_every_categorical_combination_returns_probability(self) -> None:
        option_values = [
            [float(raw) for raw in FEATURE_INFO[feature]["options"]]
            for feature in CATEGORICAL_FEATURES
        ]
        rows = []
        for combination in itertools.product(*option_values):
            values = dict(self.defaults)
            values.update(dict(zip(CATEGORICAL_FEATURES, combination)))
            rows.append(values)

        frame = pd.DataFrame(rows, columns=FEATURES)
        probabilities = self.model.predict_proba(frame)[:, 1]
        self.assertEqual(len(probabilities), 48)
        for probability in probabilities:
            self.assert_valid_probability(float(probability))

    def test_continuous_boundaries_return_probability(self) -> None:
        for feature in CONTINUOUS_FEATURES:
            info = FEATURE_INFO[feature]
            for value in (info["min"], info["default"], info["max"]):
                values = dict(self.defaults)
                values[feature] = float(value)
                self.assert_valid_probability(predict_probability(self.model, values))

    def test_supplemental_questions_are_explicitly_context_only(self) -> None:
        baseline = predict_probability(self.model, self.defaults)
        values = dict(self.defaults)
        values.update(SUPPLEMENTAL_FIELDS)
        self.assertAlmostEqual(
            predict_probability(self.model, values),
            baseline,
            places=12,
        )

    def test_filtered_fields_are_not_model_features(self) -> None:
        for excluded in ("fbs", "slope", "ca", "thal"):
            self.assertNotIn(excluded, FEATURES)
            self.assertNotIn(excluded, FEATURE_INFO)


if __name__ == "__main__":
    unittest.main()
