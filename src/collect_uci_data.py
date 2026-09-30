"""Download and harmonize the four public UCI Heart Disease files.

The source files use the same classic 14-column layout, but several test
fields are missing for most patients in one or more sources. This module
keeps only the fields selected in settings.FEATURES and preserves missing
values for the training pipeline's documented imputers.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .settings import (
    DATA_DIR,
    DATASET_SOURCES_PATH,
    EXPANDED_RAW_DATA_PATH,
    FEATURE_FILTER_REPORT_PATH,
    FEATURES,
    TARGET_COLUMN,
    UCI_SOURCE_COLUMN_NAMES,
    UCI_SOURCE_URLS,
)


INVALID_ZERO_FIELDS = ["trestbps", "chol", "thalach"]
EXCLUDED_FEATURE_REASONS = {
    "fbs": "Loại khỏi mô hình mở rộng vì thiếu dữ liệu quá nhiều ở nguồn Switzerland.",
    "slope": "Loại khỏi mô hình mở rộng vì thiếu dữ liệu quá nhiều ở nguồn Hungary.",
    "ca": "Loại khỏi mô hình mở rộng vì thiếu dữ liệu quá nhiều ở nguồn VA Long Beach.",
    "thal": "Loại khỏi mô hình mở rộng vì thiếu dữ liệu quá nhiều ở nguồn Hungary và VA Long Beach.",
}


def source_path(source_name: str) -> Path:
    return DATA_DIR / f"uci_{source_name}.data"


def download_sources() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for source_name, url in UCI_SOURCE_URLS.items():
        path = source_path(source_name)
        if not path.exists() or path.stat().st_size == 0:
            urlretrieve(url, path)
        paths[source_name] = path
    return paths


def read_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=UCI_SOURCE_COLUMN_NAMES,
        na_values="?",
    )
    for column in UCI_SOURCE_COLUMN_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Một số bản ghi cũ dùng 0 để biểu diễn số đo không ghi nhận.
    for column in INVALID_ZERO_FIELDS:
        frame[column] = frame[column].mask(frame[column] <= 0)
    frame["source"] = path.stem.removeprefix("uci_")
    return frame


def build_expanded_dataset(paths: dict[str, Path]) -> pd.DataFrame:
    frames = [read_source(path) for path in paths.values()]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["target"].notna()].copy()
    # Raw output vẫn dùng mã đích gốc; data_utils sẽ tạo target_binary.
    selected = combined[FEATURES + ["target"]].copy()
    selected.to_csv(EXPANDED_RAW_DATA_PATH, index=False, header=False, na_rep="?")
    return combined


def write_quality_reports(combined: pd.DataFrame) -> None:
    source_rows = []
    for source_name, frame in combined.groupby("source", sort=False):
        source_rows.append(
            {
                "source": source_name,
                "rows": int(len(frame)),
                "positive_original": int((frame["target"] > 0).sum()),
                "negative_original": int((frame["target"] == 0).sum()),
                "missing_retained_fields": int(frame[FEATURES].isna().sum().sum()),
                "missing_all_source_fields": int(frame[UCI_SOURCE_COLUMN_NAMES].isna().sum().sum()),
            }
        )
    pd.DataFrame(source_rows).to_csv(DATASET_SOURCES_PATH, index=False)

    rows = []
    for feature in UCI_SOURCE_COLUMN_NAMES[:-1]:
        source_rates = {
            source: float(frame[feature].isna().mean())
            for source, frame in combined.groupby("source", sort=False)
        }
        included = feature in FEATURES
        rows.append(
            {
                "feature": feature,
                "included_in_model": included,
                "overall_missing_rate": float(combined[feature].isna().mean()),
                "max_source_missing_rate": max(source_rates.values()),
                "missing_rate_by_source": "; ".join(
                    f"{source}={rate:.1%}" for source, rate in source_rates.items()
                ),
                "reason": (
                    "Giữ lại: biến có cùng ý nghĩa ở bốn nguồn; pipeline sẽ nội suy giá trị thiếu."
                    if included
                    else EXCLUDED_FEATURE_REASONS.get(feature, "Không được chọn trong bộ biến hài hòa.")
                ),
            }
        )
    pd.DataFrame(rows).to_csv(FEATURE_FILTER_REPORT_PATH, index=False)


def main() -> None:
    paths = download_sources()
    combined = build_expanded_dataset(paths)
    write_quality_reports(combined)
    print(f"Collected {len(combined)} rows from {len(paths)} UCI sources.")
    print(f"Saved harmonized raw data to {EXPANDED_RAW_DATA_PATH}")
    print(f"Saved feature filter report to {FEATURE_FILTER_REPORT_PATH}")


if __name__ == "__main__":
    main()
