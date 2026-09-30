from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT_DIR / "notebooks"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


EDA_NOTEBOOK = notebook(
    [
        markdown_cell(
            """
            # 01 - Khám phá dữ liệu bệnh tim

            Notebook này dùng để trình bày rõ dữ liệu đầu vào, nhãn mục tiêu, giá trị thiếu và các tín hiệu ban đầu liên quan tới nguy cơ mắc bệnh tim.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            ROOT = Path.cwd()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Image, display

            from src.data_utils import prepare_dataset
            from src.settings import FEATURES, FEATURE_INFO, FIGURES_DIR, TARGET_COLUMN
            """
        ),
        code_cell(
            """
            df = prepare_dataset()
            print(f"So mau: {len(df)}")
            print(f"So dac trung: {len(FEATURES)}")
            df.head()
            """
        ),
        code_cell(
            """
            target_summary = (
                df[TARGET_COLUMN]
                .map({0: "Khong nguy co", 1: "Co nguy co"})
                .value_counts()
                .rename_axis("nhom")
                .reset_index(name="so_mau")
            )
            target_summary
            """
        ),
        code_cell(
            """
            missing = df[FEATURES].isna().sum().rename("missing").reset_index()
            missing = missing.rename(columns={"index": "feature"})
            missing[missing["missing"] > 0]
            """
        ),
        code_cell(
            """
            continuous_features = [f for f in FEATURES if FEATURE_INFO[f]["kind"] == "continuous"]
            (
                df.groupby(TARGET_COLUMN)[continuous_features]
                .mean()
                .rename(index={0: "Khong nguy co", 1: "Co nguy co"})
                .T
                .round(2)
            )
            """
        ),
        code_cell(
            """
            for name in [
                "target_distribution.png",
                "feature_distributions.png",
                "correlation_heatmap.png",
                "group_mean_comparison.png",
            ]:
                path = FIGURES_DIR / name
                if path.exists():
                    display(Image(filename=str(path)))
            """
        ),
        markdown_cell(
            """
            ## Nhận xét EDA ngắn

            Các biến như kiểu đau ngực, ST depression, nhịp tim tối đa, đau thắt ngực khi vận động, số mạch máu chính và thalassemia thường có khác biệt rõ giữa hai nhóm. Đây là cơ sở để tiếp tục xây dựng mô hình phân loại và phân tích yếu tố ảnh hưởng.
            """
        ),
    ]
)


MODELING_NOTEBOOK = notebook(
    [
        markdown_cell(
            """
            # 02 - Huấn luyện, đánh giá và giải thích mô hình

            Notebook này tóm tắt quy trình modeling của project: Pipeline chống data leakage, GridSearchCV 5-fold, chọn mô hình theo tiêu chí ưu tiên Recall, calibration xác suất, tối ưu ngưỡng và giải thích yếu tố ảnh hưởng.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import json

            ROOT = Path.cwd()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import joblib
            import pandas as pd
            from IPython.display import Image, display

            from src.predict import predict_probability
            from src.settings import (
                FEATURES,
                FEATURE_INFO,
                FIGURES_DIR,
                FEATURE_IMPORTANCE_PATH,
                METADATA_PATH,
                METRICS_PATH,
                MODEL_PATH,
                THRESHOLD_ANALYSIS_PATH,
            )
            """
        ),
        code_cell(
            """
            metrics = pd.read_csv(METRICS_PATH)
            metrics[
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
            ].round(3)
            """
        ),
        code_cell(
            """
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            print("Mo hinh trien khai:", metadata["deployed_model_name"])
            print("Nguong quyet dinh:", round(metadata["decision_threshold"], 3))
            pd.Series(metadata["test_metrics"]).round(3)
            """
        ),
        code_cell(
            """
            threshold_table = pd.read_csv(THRESHOLD_ANALYSIS_PATH)
            (
                threshold_table
                .sort_values(["f2_score", "recall", "precision"], ascending=False)
                .head(10)[["threshold", "precision", "recall", "specificity", "f2_score"]]
                .round(3)
            )
            """
        ),
        code_cell(
            """
            for name in [
                "model_comparison.png",
                "confusion_matrix.png",
                "roc_curve.png",
                "precision_recall_curve.png",
                "threshold_tuning.png",
                "calibration_curve.png",
            ]:
                path = FIGURES_DIR / name
                if path.exists():
                    display(Image(filename=str(path)))
            """
        ),
        code_cell(
            """
            importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            importance.head(10).round(4)
            """
        ),
        code_cell(
            """
            for name in ["feature_importance.png", "partial_dependence.png", "shap_summary.png"]:
                path = FIGURES_DIR / name
                if path.exists():
                    display(Image(filename=str(path)))
            """
        ),
        code_cell(
            """
            model = joblib.load(MODEL_PATH)
            sample = {feature: float(FEATURE_INFO[feature]["default"]) for feature in FEATURES}
            probability = predict_probability(model, sample)
            print(f"Xac suat nguy co cua ho so mac dinh: {probability:.3f}")
            print("Vuot nguong quyet dinh:", probability >= metadata["decision_threshold"])
            """
        ),
        markdown_cell(
            """
            ## Điểm mạnh để trình bày

            Mô hình không được chọn bằng may mắn trên test set. Project dùng cross-validation trong training pool để chọn mô hình, sau đó calibration và tối ưu ngưỡng theo bài toán y tế. Test set được giữ lại đến bước cuối nên kết quả đánh giá khách quan hơn.
            """
        ),
    ]
)


def write_notebook(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "01_eda.ipynb", EDA_NOTEBOOK)
    write_notebook(NOTEBOOK_DIR / "02_modeling.ipynb", MODELING_NOTEBOOK)
    print(f"Saved notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
