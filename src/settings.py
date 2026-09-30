from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EXPANDED_RAW_DATA_PATH = DATA_DIR / "heart_uci_expanded_raw.csv"
RAW_DATA_PATH = EXPANDED_RAW_DATA_PATH
PROCESSED_DATA_PATH = DATA_DIR / "heart_uci_expanded_processed.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"
PATIENT_RECORDS_PATH = DATA_DIR / "patient_records.csv"
DATASET_SOURCES_PATH = DATA_DIR / "uci_dataset_sources.csv"
FEATURE_FILTER_REPORT_PATH = DATA_DIR / "feature_filter_report.csv"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
RAW_BEST_MODEL_PATH = MODEL_DIR / "best_raw_model.joblib"
ALL_MODELS_PATH = MODEL_DIR / "trained_models.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
CV_RESULTS_PATH = REPORTS_DIR / "cv_grid_search_results.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
INTERNAL_IMPORTANCE_PATH = REPORTS_DIR / "model_internal_importance.csv"
THRESHOLD_ANALYSIS_PATH = REPORTS_DIR / "threshold_analysis.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.csv"
REPORT_PATH = REPORTS_DIR / "bao_cao_tom_tat.md"
SLIDE_OUTLINE_PATH = REPORTS_DIR / "slide_outline.md"

UCI_SOURCE_COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Chỉ giữ các trường có cùng ý nghĩa và đủ độ phủ trong cả bốn nguồn UCI.
# fbs, slope, ca và thal bị loại vì thiếu dữ liệu quá nhiều ở ít nhất một nguồn.
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]
COLUMN_NAMES = FEATURES + ["target"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "exang"]
TARGET_COLUMN = "target_binary"

UCI_SOURCE_URLS = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}
DATASET_URL = UCI_SOURCE_URLS["cleveland"]
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
DATASET_CITATION = (
    "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). "
    "Heart Disease [Dataset; Cleveland, Hungary, Switzerland and VA Long Beach]. "
    "UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C52P4X"
)

FEATURE_INFO = {
    "age": {
        "label_vi": "Tuổi",
        "ui_label": "Tuổi (năm)",
        "unit": "năm",
        "description_vi": "Tuổi của bệnh nhân",
        "min": 20,
        "max": 85,
        "default": 54,
        "step": 1,
        "kind": "continuous",
    },
    "trestbps": {
        "label_vi": "Huyết áp tâm thu (số trên)",
        "ui_label": "Huyết áp tâm thu (số trên) lúc nghỉ (mmHg)",
        "unit": "mmHg",
        "description_vi": "Số trên của huyết áp khi nhập viện hoặc lúc nghỉ; bộ dữ liệu không có số dưới.",
        "min": 80,
        "max": 210,
        "default": 130,
        "step": 1,
        "kind": "continuous",
    },
    "chol": {
        "label_vi": "Cholesterol toàn phần",
        "ui_label": "Cholesterol toàn phần trong máu (mg/dL)",
        "unit": "mg/dL",
        "description_vi": "Lượng cholesterol toàn phần trong máu; nên xem cùng LDL, HDL và triglyceride.",
        "min": 120,
        "max": 610,
        "default": 240,
        "step": 1,
        "kind": "continuous",
    },
    "thalach": {
        "label_vi": "Nhịp tim cao nhất khi vận động",
        "ui_label": "Nhịp tim tối đa đo trong nghiệm pháp vận động (bpm)",
        "unit": "bpm",
        "description_vi": "Nhịp tim cao nhất ghi được trong nghiệm pháp vận động; không tự ước tính nếu chưa có phiếu kết quả.",
        "min": 60,
        "max": 220,
        "default": 140,
        "step": 1,
        "kind": "continuous",
    },
    "oldpeak": {
        "label_vi": "Mức chênh đoạn ST khi vận động",
        "ui_label": "Mức thay đổi đoạn ST khi vận động",
        "unit": "",
        "description_vi": "Độ chênh ST do vận động so với lúc nghỉ",
        "min": -3.0,
        "max": 6.5,
        "default": 0.5,
        "step": 0.1,
        "kind": "continuous",
    },
    "sex": {
        "label_vi": "Giới tính",
        "ui_label": "Giới tính",
        "kind": "categorical",
        "default": 1.0,
        "options": {"0.0": "Nữ", "1.0": "Nam"},
    },
    "cp": {
        "label_vi": "Kiểu đau ngực",
        "ui_label": "Kiểu đau ngực",
        "kind": "categorical",
        "default": 4.0,
        "options": {
            "1.0": "Điển hình (3 dấu hiệu)",
            "2.0": "Không điển hình (2 dấu hiệu)",
            "3.0": "Không giống kiểu đau thắt ngực điển hình (0–1 dấu hiệu)",
            "4.0": "Không triệu chứng rõ",
        },
    },
    "restecg": {
        "label_vi": "Điện tâm đồ lúc nghỉ",
        "ui_label": "Điện tâm đồ lúc nghỉ",
        "kind": "categorical",
        "default": 0.0,
        "options": {
            "0.0": "Bình thường",
            "1.0": "Bất thường đoạn ST hoặc sóng T",
            "2.0": "Dấu hiệu phì đại thất trái",
        },
    },
    "exang": {
        "label_vi": "Đau ngực khi gắng sức",
        "ui_label": "Đau ngực khi gắng sức hoặc vận động",
        "kind": "categorical",
        "default": 0.0,
        "options": {"0.0": "Không", "1.0": "Có"},
    },
}

RISK_LEVELS = [
    {
        "name": "Mức dự đoán thấp",
        "min": 0.0,
        "max": 0.4,
        "color": "#16A34A",
        "advice": "Nên duy trì lối sống lành mạnh và theo dõi sức khỏe định kỳ.",
    },
    {
        "name": "Mức dự đoán trung bình",
        "min": 0.4,
        "max": 0.7,
        "color": "#D97706",
        "advice": "Nên kiểm tra lại chỉ số và trao đổi với nhân viên y tế nếu có triệu chứng.",
    },
    {
        "name": "Mức dự đoán cao",
        "min": 0.7,
        "max": 1.01,
        "color": "#DC2626",
        "advice": "Nên sắp xếp thăm khám bác sĩ chuyên khoa tim mạch để được đánh giá chính xác.",
    },
]
