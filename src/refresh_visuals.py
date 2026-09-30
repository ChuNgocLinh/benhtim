from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_utils import prepare_dataset, save_data_dictionary, split_features_target
from .settings import (
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_BEST_MODEL_PATH,
    REPORTS_DIR,
    TEST_PREDICTIONS_PATH,
    THRESHOLD_ANALYSIS_PATH,
)
from .train_model import (
    RANDOM_STATE,
    TEST_SIZE,
    bootstrap_metric_intervals,
    save_calibration_plot,
    save_eda_plots,
    save_model_plots,
    save_partial_dependence_plot,
    save_permutation_importance,
    save_shap_plot,
    save_threshold_plot,
    write_report,
    write_slide_outline,
)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset()
    save_data_dictionary()
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_PATH)
    threshold_table = pd.read_csv(THRESHOLD_ANALYSIS_PATH)
    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

    final_model = joblib.load(MODEL_PATH)
    raw_best_model = joblib.load(RAW_BEST_MODEL_PATH)

    probabilities = test_predictions["probability"].to_numpy()
    threshold = float(metadata["decision_threshold"])
    metadata["test_metric_intervals"] = bootstrap_metric_intervals(
        y_test,
        probabilities,
        threshold,
    )
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_eda_plots(df)
    save_model_plots(metrics, X_test, y_test, threshold, probabilities)
    save_threshold_plot(threshold_table, threshold)
    save_calibration_plot(y_test, probabilities)
    feature_importance = save_permutation_importance(final_model, X_test, y_test)
    save_partial_dependence_plot(final_model, X_test, feature_importance)

    shap_model_name = str(metadata["best_model_name"]) if hasattr(raw_best_model.named_steps["model"], "feature_importances_") else None
    if shap_model_name:
        save_shap_plot(shap_model_name, raw_best_model, X_test)

    write_report(
        df,
        metrics,
        str(metadata["best_model_name"]),
        metadata["test_metrics"],
        metadata["threshold_tuning"],
        feature_importance,
        shap_model_name,
    )
    write_slide_outline(str(metadata["best_model_name"]))
    print("Refreshed visuals and reports.")


if __name__ == "__main__":
    main()
