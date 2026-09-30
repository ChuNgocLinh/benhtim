from __future__ import annotations

import pandas as pd

from .settings import (
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    CONTINUOUS_FEATURES,
    DATA_DICTIONARY_PATH,
    FEATURE_INFO,
    FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMN_NAMES, na_values="?")
    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    df["target_original"] = df["target"]
    df[TARGET_COLUMN] = (df["target"] > 0).astype(int)
    return df[FEATURES + ["target_original", TARGET_COLUMN]]


def save_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def save_data_dictionary() -> None:
    rows = []
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        rows.append(
            {
                "feature": feature,
                "label_vi": info["label_vi"],
                "ui_label": info["ui_label"],
                "kind": info["kind"],
                "unit": info.get("unit", ""),
                "description_vi": info.get("description_vi", ""),
                "options": "; ".join(
                    f"{key}: {value}" for key, value in info.get("options", {}).items()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DICTIONARY_PATH, index=False)


def split_features_target(df: pd.DataFrame):
    return df[FEATURES].copy(), df[TARGET_COLUMN].copy()


def reference_values(df: pd.DataFrame) -> dict[str, float]:
    refs: dict[str, float] = {}
    for feature in CONTINUOUS_FEATURES:
        refs[feature] = float(df[feature].median())
    for feature in CATEGORICAL_FEATURES:
        mode = df[feature].mode(dropna=True)
        refs[feature] = float(mode.iloc[0] if not mode.empty else FEATURE_INFO[feature]["default"])
    return refs


def label_for(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("label_vi", feature)

