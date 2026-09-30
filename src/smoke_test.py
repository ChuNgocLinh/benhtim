from __future__ import annotations

import sys

from .predict import explain_by_reference_delta, load_artifacts, predict_probability, risk_level
from .settings import FEATURE_INFO, FEATURES


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model, metadata = load_artifacts()
    sample = {feature: FEATURE_INFO[feature]["default"] for feature in FEATURES}
    probability = predict_probability(model, sample)
    level = risk_level(probability)
    explanation = explain_by_reference_delta(model, sample, metadata["reference_values"])
    assert 0.0 <= probability <= 1.0
    assert not explanation.empty
    print(f"Smoke test passed. Probability={probability:.3f}, level={level['name']}")


if __name__ == "__main__":
    main()
