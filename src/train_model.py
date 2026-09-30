from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data_utils import (
    label_for,
    prepare_dataset,
    reference_values,
    save_data_dictionary,
    save_processed_dataset,
    split_features_target,
)
from .settings import (
    ALL_MODELS_PATH,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    CV_RESULTS_PATH,
    DATASET_CITATION,
    DATASET_PAGE,
    EDA_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_INFO,
    FEATURES,
    FIGURES_DIR,
    INTERNAL_IMPORTANCE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RAW_BEST_MODEL_PATH,
    REPORTS_DIR,
    REPORT_PATH,
    SLIDE_OUTLINE_PATH,
    TARGET_COLUMN,
    TEST_PREDICTIONS_PATH,
    THRESHOLD_ANALYSIS_PATH,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_RECALL_TARGET = 0.90
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
}

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


MODEL_NAME_VI = {
    "Logistic Regression": "Hồi quy Logistic",
    "K-Nearest Neighbors": "K láng giềng gần nhất",
    "Support Vector Machine": "Máy vector hỗ trợ",
    "Random Forest": "Rừng ngẫu nhiên",
    "Gradient Boosting": "Tăng cường Gradient",
    "XGBoost": "XGBoost",
}

METRIC_NAME_VI = {
    "accuracy": "Độ chính xác tổng thể",
    "precision": "Độ chính xác cảnh báo",
    "recall": "Độ nhạy",
    "f1_score": "Điểm F1",
    "roc_auc": "AUC ROC",
}


def model_name_vi(name: str) -> str:
    return MODEL_NAME_VI.get(name, name)


def make_preprocessor() -> ColumnTransformer:
    continuous_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", continuous_pipe, CONTINUOUS_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )


def model_spaces() -> dict:
    spaces = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            {
                "model__C": [0.05, 0.1, 0.5, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "K-Nearest Neighbors": (
            make_pipeline(KNeighborsClassifier()),
            {
                "model__n_neighbors": [5, 7, 9, 11, 13, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Support Vector Machine": (
            make_pipeline(SVC(random_state=RANDOM_STATE)),
            {
                "model__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            {
                "model__n_estimators": [250, 500],
                "model__max_depth": [None, 4, 7],
                "model__min_samples_leaf": [1, 3, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.03, 0.06, 0.10],
                "model__max_depth": [2, 3],
            },
        ),
    }
    if HAS_XGBOOST:
        spaces["XGBoost"] = (
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            {
                "model__n_estimators": [120, 220],
                "model__learning_rate": [0.03, 0.06, 0.10],
                "model__max_depth": [2, 3],
                "model__subsample": [0.85],
                "model__colsample_bytree": [0.85],
            },
        )
    return spaces


def clinical_selection_score(recall: float, roc_auc: float, f1: float) -> float:
    return 0.45 * recall + 0.35 * roc_auc + 0.20 * f1


def clinical_refit(cv_results: dict) -> int:
    score = clinical_selection_score(
        np.asarray(cv_results["mean_test_recall"], dtype=float),
        np.asarray(cv_results["mean_test_roc_auc"], dtype=float),
        np.asarray(cv_results["mean_test_f1_score"], dtype=float),
    )
    return int(np.nanargmax(score))


def scores_for_ranking(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_default_model(model, X_test, y_test) -> dict:
    prediction = model.predict(X_test)
    scores = scores_for_ranking(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1_score": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "average_precision": average_precision_score(y_test, scores),
    }


def evaluate_probabilities(y_true, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, prediction, zero_division=0),
        "f2_score": fbeta_score(y_true, prediction, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def bootstrap_metric_intervals(
    y_true,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Estimate uncertainty of holdout metrics without changing the test labels."""
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "roc_auc": [],
        "brier_score": [],
    }
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        sampled_y = y_array[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_metrics = evaluate_probabilities(
            sampled_y,
            probability_array[indices],
            threshold,
        )
        for metric in samples:
            samples[metric].append(float(sampled_metrics[metric]))

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float]] = {}
    for metric, values in samples.items():
        if not values:
            continue
        intervals[metric] = {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "median": float(np.quantile(values, 0.5)),
        }
    return intervals


def train_models(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trained = {}
    metric_rows = []
    grid_rows = []
    for name, (pipeline, params) in model_spaces().items():
        print(f"Training {name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            refit=clinical_refit,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        trained[name] = best_model
        best_index = int(search.best_index_)
        cv_metrics = {
            metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
            for metric in SCORING
        }
        cv_stds = {
            f"{metric}_std": float(search.cv_results_[f"std_test_{metric}"][best_index])
            for metric in SCORING
        }
        default_test = evaluate_default_model(best_model, X_test, y_test)
        selection_score = clinical_selection_score(
            cv_metrics["recall"], cv_metrics["roc_auc"], cv_metrics["f1_score"]
        )
        row = {
            "model": name,
            "selected_by_cv": False,
            "cv_accuracy": cv_metrics["accuracy"],
            "cv_precision": cv_metrics["precision"],
            "cv_recall": cv_metrics["recall"],
            "cv_f1_score": cv_metrics["f1_score"],
            "cv_roc_auc": cv_metrics["roc_auc"],
            "cv_selection_score": selection_score,
            "cv_accuracy_std": cv_stds["accuracy_std"],
            "cv_precision_std": cv_stds["precision_std"],
            "cv_recall_std": cv_stds["recall_std"],
            "cv_f1_score_std": cv_stds["f1_score_std"],
            "cv_roc_auc_std": cv_stds["roc_auc_std"],
            "accuracy": default_test["accuracy"],
            "precision": default_test["precision"],
            "recall": default_test["recall"],
            "f1_score": default_test["f1_score"],
            "roc_auc": default_test["roc_auc"],
            "average_precision": default_test["average_precision"],
            "selection_score": selection_score,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        }
        metric_rows.append(row)

        result_frame = pd.DataFrame(search.cv_results_)
        result_frame.insert(0, "model", name)
        grid_rows.append(result_frame)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["cv_selection_score", "cv_recall", "cv_roc_auc"], ascending=False
    )
    metrics.loc[metrics.index[0], "selected_by_cv"] = True
    grid_results = pd.concat(grid_rows, ignore_index=True)
    return trained, metrics, grid_results


def tune_threshold(y_true, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict]:
    thresholds = np.round(np.linspace(0.05, 0.95, 181), 3)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_probabilities(y_true, probabilities, float(threshold))
        rows.append(metrics)
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= MIN_RECALL_TARGET].copy()
    if eligible.empty:
        chosen = table.sort_values(["f2_score", "recall", "precision"], ascending=False).iloc[0]
        rule = "Tối đa F2-score vì không có ngưỡng nào đạt recall mục tiêu"
    else:
        chosen = eligible.sort_values(["f2_score", "precision", "specificity"], ascending=False).iloc[0]
        rule = f"Tối đa F2-score trong các ngưỡng có recall >= {MIN_RECALL_TARGET:.2f}"
    summary = {
        "threshold": float(chosen["threshold"]),
        "rule": rule,
        "validation_recall": float(chosen["recall"]),
        "validation_precision": float(chosen["precision"]),
        "validation_specificity": float(chosen["specificity"]),
        "validation_f2_score": float(chosen["f2_score"]),
    }
    return table, summary


def calibrated_oof_probabilities(base_model, X_train, y_train) -> np.ndarray:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    probabilities = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )
    return probabilities[:, 1]


def fit_calibrated_model(base_model, X_train, y_train):
    calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def clean_transformed_name(name: str) -> str:
    return name.split("__", 1)[-1]


def original_feature_from_transformed(name: str) -> str:
    clean = clean_transformed_name(name)
    for feature in sorted(FEATURES, key=len, reverse=True):
        if clean == feature or clean.startswith(f"{feature}_"):
            return feature
    return clean


def internal_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [clean_transformed_name(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(estimator, "feature_importances_"):
        raw_importance = np.abs(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        raw_importance = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame()
    result = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "original_feature": [original_feature_from_transformed(name) for name in feature_names],
            "importance": raw_importance,
        }
    )
    grouped = (
        result.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    total = grouped["importance"].sum()
    grouped["importance_normalized"] = grouped["importance"] / total if total else 0
    grouped["label_vi"] = grouped["original_feature"].map(label_for)
    return grouped


def save_eda_summary(df: pd.DataFrame) -> None:
    rows = []
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        row = {
            "feature": feature,
            "label_vi": info["label_vi"],
            "kind": info["kind"],
            "missing": int(df[feature].isna().sum()),
        }
        if info["kind"] == "continuous":
            row.update(
                {
                    "mean_no_risk": df.loc[df[TARGET_COLUMN] == 0, feature].mean(),
                    "mean_risk": df.loc[df[TARGET_COLUMN] == 1, feature].mean(),
                    "median_no_risk": df.loc[df[TARGET_COLUMN] == 0, feature].median(),
                    "median_risk": df.loc[df[TARGET_COLUMN] == 1, feature].median(),
                }
            )
        else:
            row.update(
                {
                    "mean_no_risk": np.nan,
                    "mean_risk": np.nan,
                    "median_no_risk": np.nan,
                    "median_risk": np.nan,
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(EDA_SUMMARY_PATH, index=False)


def save_eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="deep")

    target_counts = df[TARGET_COLUMN].map({0: "Không có bệnh", 1: "Có bệnh"}).value_counts()
    target_frame = pd.DataFrame({"target": target_counts.index, "count": target_counts.values})
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(
        data=target_frame,
        x="target",
        y="count",
        hue="target",
        palette=["#0F766E", "#DC2626"],
        legend=False,
    )
    ax.set_title("Phân bố lớp mục tiêu")
    ax.set_xlabel("")
    ax.set_ylabel("Số mẫu")
    for index, value in enumerate(target_counts.values):
        ax.text(index, value + 2, str(value), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=170)
    plt.close()

    corr = df[FEATURES + [TARGET_COLUMN]].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.4, linecolor="#E5E7EB")
    plt.title("Ma trận tương quan giữa các biến")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=170)
    plt.close()

    selected = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for index, feature in enumerate(selected):
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[index],
            palette=["#0F766E", "#DC2626"],
        )
        axes[index].set_title(FEATURE_INFO[feature]["label_vi"])
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_distributions.png", dpi=170)
    plt.close()

    means = (
        df.groupby(TARGET_COLUMN)[CONTINUOUS_FEATURES]
        .mean()
        .rename(index={0: "Không có bệnh", 1: "Có bệnh"})
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )
    means["label_vi"] = means["feature"].map(label_for)
    plot_frame = means.melt(
        id_vars=["feature", "label_vi"],
        value_vars=["Không có bệnh", "Có bệnh"],
        var_name="group",
        value_name="mean_value",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_frame, x="label_vi", y="mean_value", hue="group", palette=["#0F766E", "#DC2626"])
    plt.title("Giá trị trung bình của biến liên tục theo nhóm")
    plt.xlabel("")
    plt.ylabel("Giá trị trung bình")
    plt.xticks(rotation=16, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "group_mean_comparison.png", dpi=170)
    plt.close()


def save_model_plots(
    metrics: pd.DataFrame,
    X_test,
    y_test,
    threshold: float,
    probabilities: np.ndarray,
) -> None:
    plot_metrics = metrics.melt(
        id_vars="model",
        value_vars=["cv_accuracy", "cv_precision", "cv_recall", "cv_f1_score", "cv_roc_auc"],
        var_name="metric",
        value_name="score",
    )
    plot_metrics["model"] = plot_metrics["model"].map(model_name_vi)
    plot_metrics["metric"] = (
        plot_metrics["metric"]
        .str.replace("cv_", "", regex=False)
        .map(METRIC_NAME_VI)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_metrics, x="model", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("So sánh mô hình bằng kiểm định chéo")
    plt.xlabel("")
    plt.ylabel("Điểm số")
    plt.xticks(rotation=18, ha="right")
    plt.legend(loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=170)
    plt.close()

    prediction = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Không có bệnh", "Có bệnh"],
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Ma trận nhầm lẫn với ngưỡng tối ưu")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=170)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#0F766E", linewidth=2.4, label=f"AUC ROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--")
    plt.xlabel("Tỷ lệ báo nhầm")
    plt.ylabel("Tỷ lệ phát hiện đúng")
    plt.title("Đường cong ROC của mô hình đã hiệu chỉnh")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=170)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="#2563EB", linewidth=2.4, label=f"Độ chính xác trung bình = {ap:.3f}")
    plt.xlabel("Độ nhạy")
    plt.ylabel("Độ chính xác cảnh báo")
    plt.title("Đường cong độ chính xác - độ nhạy")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=170)
    plt.close()


def save_threshold_plot(threshold_table: pd.DataFrame, threshold: float) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_table["threshold"], threshold_table["precision"], label="Độ chính xác cảnh báo", color="#2563EB")
    plt.plot(threshold_table["threshold"], threshold_table["recall"], label="Độ nhạy", color="#DC2626")
    plt.plot(threshold_table["threshold"], threshold_table["f2_score"], label="Điểm F2", color="#0F766E")
    plt.axvline(threshold, color="#111827", linestyle="--", linewidth=1.6, label=f"Ngưỡng chọn = {threshold:.2f}")
    plt.ylim(0, 1.04)
    plt.xlabel("Ngưỡng phân loại")
    plt.ylabel("Điểm số")
    plt.title("Tối ưu ngưỡng dự đoán theo điểm F2 và độ nhạy")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tuning.png", dpi=170)
    plt.close()


def save_calibration_plot(y_test, probabilities: np.ndarray) -> None:
    prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=6, strategy="quantile")
    plt.figure(figsize=(6.5, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2.2, color="#0F766E", label="Mô hình")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Lý tưởng")
    plt.xlabel("Xác suất dự đoán trung bình")
    plt.ylabel("Tỷ lệ dương tính thực tế")
    plt.title("Đường hiệu chỉnh xác suất")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=170)
    plt.close()


def save_permutation_importance(model, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "label_vi": [label_for(feature) for feature in FEATURES],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    top = importance.head(10).sort_values("importance_mean", ascending=True)
    plt.figure(figsize=(8, 5.5))
    colors = ["#0F766E" if value >= 0 else "#94A3B8" for value in top["importance_mean"]]
    plt.barh(top["label_vi"], top["importance_mean"], color=colors)
    plt.xlabel("Mức giảm ROC-AUC khi xáo trộn biến")
    plt.title("Top yếu tố ảnh hưởng theo phương pháp hoán vị")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=170)
    plt.close()
    return importance


def save_partial_dependence_plot(model, X_test: pd.DataFrame, importance: pd.DataFrame) -> str | None:
    candidate_features = [
        feature
        for feature in importance["feature"].tolist()
        if feature in CONTINUOUS_FEATURES
    ][:4]
    if len(candidate_features) < 2:
        candidate_features = ["oldpeak", "thalach", "age", "trestbps"]
    try:
        fig, ax = plt.subplots(figsize=(11, 7))
        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            features=candidate_features,
            response_method="predict_proba",
            target=1,
            ax=ax,
        )
        flat_axes = np.ravel(ax)
        for axis, feature in zip(flat_axes, candidate_features):
            axis.set_title(label_for(feature))
            axis.set_xlabel(label_for(feature))
            axis.set_ylabel("Xác suất dự đoán thuộc nhóm có bệnh")
        fig.suptitle("Xu hướng xác suất theo các biến liên tục quan trọng")
        fig.tight_layout()
        path = FIGURES_DIR / "partial_dependence.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return str(path)
    except Exception as exc:
        print(f"Partial dependence skipped: {exc}")
        return None


def save_shap_plot(model_name: str, model: Pipeline, X_sample: pd.DataFrame) -> str | None:
    if not HAS_SHAP:
        return None
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    try:
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(X_sample)
        feature_names = [
            label_for(original_feature_from_transformed(clean_transformed_name(name)))
            for name in preprocessor.get_feature_names_out()
        ]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            values = shap_values[1]
        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[:, :, 1]
        else:
            values = shap_values
        shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=12)
        plt.title(f"Tóm tắt SHAP - {model_name_vi(model_name)}")
        plt.tight_layout()
        path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        return str(path)
    except Exception as exc:
        print(f"SHAP plot skipped: {exc}")
        return None


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    best_name: str,
    final_metrics: dict,
    threshold_summary: dict,
    feature_importance: pd.DataFrame,
    shap_model_name: str | None,
) -> None:
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    missing = df[FEATURES].isna().sum()
    threshold_rule = (
        threshold_summary["rule"]
        .replace("F2-score", "điểm F2")
        .replace("recall", "độ nhạy")
    )
    missing_display = missing.reset_index().rename(
        columns={"index": "Yếu tố", 0: "Số giá trị thiếu"}
    )
    top_features = feature_importance.head(8)[
        ["feature", "label_vi", "importance_mean", "importance_std"]
    ].rename(
        columns={
            "feature": "Mã biến",
            "label_vi": "Yếu tố",
            "importance_mean": "Mức ảnh hưởng trung bình",
            "importance_std": "Độ dao động",
        }
    )
    display_metrics = metrics[
        [
            "model",
            "selected_by_cv",
            "cv_recall",
            "cv_f1_score",
            "cv_roc_auc",
            "cv_selection_score",
            "recall",
            "roc_auc",
        ]
    ].copy()
    display_metrics["model"] = display_metrics["model"].map(model_name_vi)
    display_metrics["selected_by_cv"] = display_metrics["selected_by_cv"].map(
        lambda value: "Có" if value else "Không"
    )
    display_metrics = display_metrics.rename(
        columns={
            "model": "Mô hình",
            "selected_by_cv": "Được chọn",
            "cv_recall": "Độ nhạy kiểm định chéo",
            "cv_f1_score": "F1 kiểm định chéo",
            "cv_roc_auc": "AUC ROC kiểm định chéo",
            "cv_selection_score": "Điểm chọn",
            "recall": "Độ nhạy kiểm tra",
            "roc_auc": "AUC ROC kiểm tra",
        }
    )
    shap_note = (
        f"Có tóm tắt SHAP cho mô hình {model_name_vi(shap_model_name)}."
        if shap_model_name
        else "Không tạo tóm tắt SHAP vì mô hình triển khai hiện tại không hỗ trợ giải thích SHAP trực tiếp."
    )
    report = f"""# Báo cáo tóm tắt: Phân loại sự hiện diện bệnh tim trong dữ liệu

## 1. Mục tiêu

Xây dựng hệ thống hỗ trợ phân loại sự hiện diện bệnh tim theo bốn nguồn UCI đã được hài hòa, đồng thời xác định những yếu tố ảnh hưởng lớn nhất đến kết quả dự đoán.

## 2. Dữ liệu

- Nguồn: UCI Machine Learning Repository - Heart Disease Dataset.
- Trích dẫn: {DATASET_CITATION}.
- Nguồn con: Cleveland, Hungary, Switzerland và VA Long Beach; các biến thiếu quá nhiều ở từng nguồn đã được lọc khỏi đầu vào.
- Số mẫu: {len(df)}.
- Số đặc trưng đầu vào: {len(FEATURES)}.
- Nhãn gốc `target`: 0 là không có bệnh, 1-4 là có bệnh theo nhãn của bộ dữ liệu. Project chuyển nhãn thành bài toán nhị phân: `0 = không có bệnh`, `1 = có bệnh`.
- Phân bố lớp: không có bệnh = {int(target_counts.get(0, 0))}, có bệnh = {int(target_counts.get(1, 0))}.

## 3. Tiền xử lý

- Chuyển ký hiệu thiếu `?` thành giá trị rỗng.
- Dùng `SimpleImputer`: trung vị cho biến liên tục, giá trị xuất hiện nhiều nhất cho biến phân loại.
- Chuẩn hóa biến liên tục bằng `StandardScaler`.
- Mã hóa one-hot cho biến phân loại bằng `OneHotEncoder`.
- Đặt toàn bộ tiền xử lý trong một đường ống xử lý để tránh rò rỉ dữ liệu từ tập kiểm tra.

Giá trị thiếu theo cột:

{markdown_table(missing_display, ["Yếu tố", "Số giá trị thiếu"])}

## 4. Quy trình đánh giá nâng cao

- Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80/20, có phân tầng để giữ tỷ lệ lớp.
- Chỉ dùng tập huấn luyện cho tìm kiếm tham số và kiểm định chéo 5 phần.
- Chọn mô hình bằng `điểm chọn = 0.45 * độ nhạy + 0.35 * AUC ROC + 0.20 * F1`.
- Tập kiểm tra chỉ được dùng một lần ở cuối để đánh giá khách quan.
- Hiệu chỉnh xác suất bằng `CalibratedClassifierCV`.
- Tối ưu ngưỡng phân loại bằng xác suất ngoài-fold trên tập huấn luyện, ưu tiên điểm F2 và độ nhạy.

## 5. So sánh mô hình

{markdown_table(display_metrics, ["Mô hình", "Được chọn", "Độ nhạy kiểm định chéo", "F1 kiểm định chéo", "AUC ROC kiểm định chéo", "Điểm chọn", "Độ nhạy kiểm tra", "AUC ROC kiểm tra"])}

Mô hình được chọn bằng kiểm định chéo: **{model_name_vi(best_name)}**.

## 6. Kết quả cuối trên tập kiểm tra

- Ngưỡng phân loại tối ưu: {final_metrics["threshold"]:.3f}.
- Độ chính xác tổng thể: {final_metrics["accuracy"]:.3f}.
- Độ chính xác cảnh báo: {final_metrics["precision"]:.3f}.
- Độ nhạy: {final_metrics["recall"]:.3f}.
- Độ đặc hiệu: {final_metrics["specificity"]:.3f}.
- Điểm F1: {final_metrics["f1_score"]:.3f}.
- Điểm F2: {final_metrics["f2_score"]:.3f}.
- AUC ROC: {final_metrics["roc_auc"]:.3f}.
- Độ chính xác trung bình: {final_metrics["average_precision"]:.3f}.
- Sai số xác suất Brier: {final_metrics["brier_score"]:.3f}.

Quy tắc chọn ngưỡng: {threshold_rule}.

## 7. Yếu tố ảnh hưởng lớn nhất

Mức ảnh hưởng theo hoán vị cho biết khi xáo trộn từng biến thì AUC ROC giảm bao nhiêu. Biến làm mô hình giảm hiệu năng nhiều hơn được xem là quan trọng hơn.

{markdown_table(top_features, ["Mã biến", "Yếu tố", "Mức ảnh hưởng trung bình", "Độ dao động"])}

Các biểu đồ bổ sung gồm đường cong ROC, đường cong độ chính xác - độ nhạy, đường hiệu chỉnh xác suất, biểu đồ tối ưu ngưỡng, biểu đồ xu hướng xác suất theo từng yếu tố và SHAP nếu mô hình hỗ trợ trực tiếp. {shap_note}

## 8. Kết luận

Hệ thống đã triển khai tiền xử lý, so sánh nhiều mô hình, chọn mô hình bằng kiểm định chéo, hiệu chỉnh xác suất, tối ưu mốc sàng lọc, đánh giá trên một tập kiểm tra nội bộ và giải thích yếu tố ảnh hưởng. Giao diện demo hiển thị xác suất dự đoán theo 3 dải tham khảo và giải thích cá nhân hóa cho từng hồ sơ.

Lưu ý: Hệ thống chỉ mang tính hỗ trợ học tập và sàng lọc tham khảo, không thay thế chẩn đoán của bác sĩ.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_slide_outline(best_name: str) -> None:
    text = f"""# Dàn ý slide thuyết trình

## Slide 1 - Tên đề tài
Xây dựng hệ thống phân loại sự hiện diện bệnh tim trong dữ liệu và phân tích yếu tố ảnh hưởng.

## Slide 2 - Bài toán
Đầu vào là {len(FEATURES)} biến đã được lọc từ bốn nguồn UCI, đầu ra là xác suất mô hình xếp hồ sơ vào nhóm có hoặc không có bệnh theo nhãn dữ liệu.

## Slide 3 - Dữ liệu
UCI Heart Disease Dataset, bốn nguồn với {len(FEATURES)} đặc trưng sau lọc, target chuyển về nhị phân.

## Slide 4 - Khám phá dữ liệu
Phân bố lớp, giá trị thiếu, heatmap tương quan và phân bố các biến liên tục.

## Slide 5 - Tiền xử lý
Điền giá trị thiếu, chuẩn hóa biến liên tục, mã hóa one-hot và gom toàn bộ vào đường ống xử lý để tránh rò rỉ dữ liệu.

## Slide 6 - Mô hình
Hồi quy Logistic, K láng giềng gần nhất, Máy vector hỗ trợ, Rừng ngẫu nhiên, Tăng cường Gradient và XGBoost.

## Slide 7 - Chọn mô hình
Tìm kiếm tham số, kiểm định chéo 5 phần và điểm chọn mô hình ưu tiên độ nhạy.

## Slide 8 - Hiệu chỉnh xác suất
Hiệu chỉnh xác suất, đường hiệu chỉnh xác suất và sai số xác suất Brier.

## Slide 9 - Tối ưu mốc sàng lọc
Đường cong độ chính xác - độ nhạy, tối ưu mốc và điểm F2 để ưu tiên độ nhạy trên dữ liệu đánh giá.

## Slide 10 - Kết quả
Mô hình được chọn: {model_name_vi(best_name)}. Trình bày độ chính xác tổng thể, độ chính xác cảnh báo, độ nhạy, độ đặc hiệu, điểm F1, điểm F2 và AUC ROC.

## Slide 11 - Giải thích mô hình
Mức ảnh hưởng theo hoán vị, SHAP nếu mô hình hỗ trợ trực tiếp và biểu đồ xu hướng xác suất theo từng yếu tố.

## Slide 12 - Demo giao diện
Nhập hồ sơ bệnh nhân, xem xác suất dự đoán, diễn giải và yếu tố ảnh hưởng; kết quả không thay thế chẩn đoán.

## Slide 13 - Kết luận
Hệ thống có khả năng dự đoán, đánh giá và giải thích. Kết quả chỉ phục vụ học tập và tham khảo.
"""
    SLIDE_OUTLINE_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset()
    save_processed_dataset(df)
    save_data_dictionary()
    save_eda_summary(df)
    save_eda_plots(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    trained, metrics, grid_results = train_models(X_train, y_train, X_test, y_test)
    metrics.to_csv(METRICS_PATH, index=False)
    grid_results.to_csv(CV_RESULTS_PATH, index=False)

    best_name = str(metrics.iloc[0]["model"])
    best_raw_model = trained[best_name]
    best_default_metrics = metrics.iloc[0].to_dict()

    print(f"Calibrating selected model: {best_name}...")
    oof_probability = calibrated_oof_probabilities(best_raw_model, X_train, y_train)
    threshold_table, threshold_summary = tune_threshold(y_train, oof_probability)
    threshold_table.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)

    final_model = fit_calibrated_model(best_raw_model, X_train, y_train)
    test_probability = final_model.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_probabilities(y_test, test_probability, threshold_summary["threshold"])
    test_prediction = (test_probability >= final_metrics["threshold"]).astype(int)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(best_raw_model, RAW_BEST_MODEL_PATH)
    joblib.dump(trained, ALL_MODELS_PATH)

    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "probability": test_probability,
            "prediction": test_prediction,
        }
    ).to_csv(TEST_PREDICTIONS_PATH, index=False)

    save_model_plots(metrics, X_test, y_test, final_metrics["threshold"], test_probability)
    save_threshold_plot(threshold_table, final_metrics["threshold"])
    save_calibration_plot(y_test, test_probability)
    feature_importance = save_permutation_importance(final_model, X_test, y_test)
    partial_dependence_path = save_partial_dependence_plot(final_model, X_test, feature_importance)

    internal = internal_importance(best_raw_model)
    if not internal.empty:
        internal.to_csv(INTERNAL_IMPORTANCE_PATH, index=False)

    shap_model_name = best_name if hasattr(best_raw_model.named_steps["model"], "feature_importances_") else None
    shap_path = save_shap_plot(shap_model_name, best_raw_model, X_test) if shap_model_name else None

    class_report = classification_report(y_test, test_prediction, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_prediction, labels=[0, 1]).tolist()
    test_metric_intervals = bootstrap_metric_intervals(
        y_test,
        test_probability,
        final_metrics["threshold"],
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_name": best_name,
        "deployed_model_name": f"Calibrated {best_name}",
        "model_selection_protocol": (
            "GridSearchCV and 5-fold cross-validation on training pool only. "
            "Final test set is held out until the final calibrated evaluation."
        ),
        "selection_rule": "0.45*cv_recall + 0.35*cv_roc_auc + 0.20*cv_f1_score",
        "decision_threshold": float(final_metrics["threshold"]),
        "threshold_tuning": threshold_summary,
        "dataset_page": DATASET_PAGE,
        "dataset_citation": DATASET_CITATION,
        "dataset_label": "UCI Heart Disease (4 nguồn đã hài hòa)",
        "dataset_sources": ["cleveland", "hungarian", "switzerland", "va"],
        "excluded_features": ["fbs", "slope", "ca", "thal"],
        "n_samples": int(len(df)),
        "n_features": int(len(FEATURES)),
        "split": {
            "train_pool": int(len(X_train)),
            "test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
        },
        "target_distribution": {
            str(key): int(value) for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "missing_values": {feature: int(value) for feature, value in df[FEATURES].isna().sum().items()},
        "features": FEATURE_INFO,
        "reference_values": reference_values(df),
        "cv_best_metrics": {
            "recall": float(best_default_metrics["cv_recall"]),
            "f1_score": float(best_default_metrics["cv_f1_score"]),
            "roc_auc": float(best_default_metrics["cv_roc_auc"]),
            "selection_score": float(best_default_metrics["cv_selection_score"]),
        },
        "test_metrics_default_threshold_before_calibration": {
            key: float(best_default_metrics[key])
            for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
        },
        "test_metrics": {
            key: float(final_metrics[key])
            for key in [
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "f2_score",
                "roc_auc",
                "average_precision",
                "brier_score",
            ]
        },
        "confusion_matrix": cm,
        "test_metric_intervals": test_metric_intervals,
        "classification_report": class_report,
        "test_probabilities": [float(value) for value in test_probability],
        "shap_model_name": shap_model_name if shap_path else None,
        "shap_summary_path": shap_path,
        "partial_dependence_path": partial_dependence_path,
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(df, metrics, best_name, final_metrics, threshold_summary, feature_importance, shap_model_name)
    write_slide_outline(best_name)

    print("\nTraining completed.")
    print(f"Best model selected by CV: {best_name}")
    print(f"Decision threshold: {final_metrics['threshold']:.3f}")
    print(
        metrics[
            ["model", "selected_by_cv", "cv_recall", "cv_f1_score", "cv_roc_auc", "cv_selection_score"]
        ].to_string(index=False)
    )
    print(
        "Final test metrics: "
        f"accuracy={final_metrics['accuracy']:.3f}, "
        f"precision={final_metrics['precision']:.3f}, "
        f"recall={final_metrics['recall']:.3f}, "
        f"roc_auc={final_metrics['roc_auc']:.3f}, "
        f"brier={final_metrics['brier_score']:.3f}"
    )
    print(f"Saved calibrated model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
