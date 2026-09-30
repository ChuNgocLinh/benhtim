from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import (
    display_value,
    explain_by_reference_delta,
    load_artifacts,
    predict_probability,
    risk_level,
)
from src.records import load_patient_records, save_patient_record
from src.settings import (
    DATASET_SOURCES_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_FILTER_REPORT_PATH,
    FEATURE_INFO,
    FEATURES,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PATIENT_RECORDS_PATH,
    THRESHOLD_ANALYSIS_PATH,
)


st.set_page_config(
    page_title="Trợ lý Tim Mạch",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #F7FAFC 0%, #EEF7F8 100%);
        color: #172033;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    html, body, .stApp, .stMarkdown, p, li {
        font-size: 16px;
        line-height: 1.55;
    }
    [data-testid="stToolbar"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenuButton"],
    [data-testid="stBaseButton-header"],
    [data-testid="stHeaderActionElements"],
    [data-testid="stStatusWidget"],
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButtonContainer"],
    [data-testid="stElementToolbarButton"],
    [data-testid="stBaseButton-elementToolbar"],
    .stDeployButton,
    #MainMenu,
    footer,
    button[title="View fullscreen"],
    button[aria-label="Fullscreen"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        display: none !important;
        height: 0 !important;
    }
    .app-hero {
        border: 1px solid #D8E7EA;
        background: linear-gradient(135deg, #FFFFFF 0%, #F5FBFA 54%, #EEF7F2 100%);
        border-radius: 8px;
        padding: 30px 34px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07);
        margin-bottom: 22px;
    }
    .clinical-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: -8px 0 20px 0;
    }
    .clinical-strip-item {
        background: #FFFFFF;
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        padding: 14px 15px;
    }
    .clinical-strip-item span {
        display: block;
        color: #64748B;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .clinical-strip-item strong {
        color: #102A43;
        font-size: 16px;
        line-height: 1.35;
    }
    .validation-summary {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 8px 0 18px 0;
    }
    .validation-card {
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 13px 14px;
    }
    .validation-card span {
        display: block;
        color: #64748B;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .validation-card strong {
        display: block;
        color: #102A43;
        font-size: 19px;
        line-height: 1.2;
    }
    .validation-card small {
        display: block;
        color: #64748B;
        font-size: 12px;
        margin-top: 5px;
        line-height: 1.35;
    }
    .validation-note {
        border-left: 4px solid #64748B;
        background: #F8FAFC;
        color: #475569;
        border-radius: 6px;
        padding: 11px 13px;
        margin: 0 0 16px 0;
        font-size: 14px;
        line-height: 1.45;
    }
    .eyebrow {
        margin: 0 0 8px 0;
        color: #0F766E;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .app-hero h1 {
        margin: 0;
        color: #102A43;
        font-size: 40px;
        line-height: 1.15;
        letter-spacing: 0;
    }
    .app-hero p {
        margin: 10px 0 0 0;
        color: #475569;
        font-size: 17px;
        max-width: 680px;
    }
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(125px, 1fr));
        gap: 10px;
        min-width: 430px;
    }
    .stat {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        background: #FFFFFF;
    }
    .stat span {
        display: block;
        color: #64748B;
        font-size: 13px;
        margin-bottom: 5px;
    }
    .stat strong {
        color: #0F172A;
        font-size: 17px;
        line-height: 1.18;
        overflow-wrap: anywhere;
    }
    .panel {
        border: 1px solid #DDE8EA;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        margin: 0 0 14px 0;
        color: #102A43;
        font-size: 23px;
        font-weight: 760;
    }
    .risk-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 8px;
        padding: 10px 14px;
        color: #FFFFFF;
        font-weight: 760;
        font-size: 17px;
        margin-bottom: 10px;
    }
    .advice {
        border-left: 4px solid #0F766E;
        background: #F0FDFA;
        color: #134E4A;
        padding: 13px 15px;
        border-radius: 6px;
        margin-top: 14px;
        font-size: 16px;
    }
    .medical-note {
        color: #475569;
        font-size: 15px;
        line-height: 1.55;
        padding-top: 8px;
    }
    .small-muted {
        color: #64748B;
        font-size: 15px;
        margin-top: -6px;
        margin-bottom: 12px;
    }
    .patient-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 4px 0 18px 0;
    }
    .patient-card {
        background: #FFFFFF;
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        padding: 14px;
        min-height: 84px;
    }
    .patient-card span {
        display: block;
        color: #64748B;
        font-size: 13px;
        margin-bottom: 5px;
    }
    .patient-card strong {
        color: #102A43;
        font-size: 19px;
        line-height: 1.2;
    }
    .input-group-title {
        border-left: 4px solid #0F766E;
        background: #FFFFFF;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 16px 0 12px 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    .input-group-title strong {
        display: block;
        color: #102A43;
        font-size: 17px;
        margin-bottom: 3px;
    }
    .input-group-title span {
        color: #475569;
        font-size: 15px;
        line-height: 1.45;
    }
    .flag-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
        margin: 8px 0 16px 0;
    }
    .flag-card {
        border-radius: 8px;
        border: 1px solid #DDE8EA;
        background: #FFFFFF;
        padding: 12px 13px;
        color: #334155;
        font-size: 15px;
        line-height: 1.45;
    }
    .flag-card strong {
        display: block;
        color: #0F766E;
        font-size: 14px;
        margin-bottom: 3px;
    }
    .flag-card.warning {
        border-color: #FED7AA;
        background: #FFF7ED;
    }
    .flag-card.warning strong {
        color: #C2410C;
    }
    .flag-card.info {
        border-color: #CFE7E3;
        background: #F7FEFB;
    }
    .flag-card.info strong {
        color: #0F766E;
    }
    .flag-card.critical {
        border-color: #FECACA;
        background: #FEF2F2;
    }
    .flag-card.critical strong {
        color: #B91C1C;
    }
    .section-band {
        border: 1px solid #D8E7EA;
        background: rgba(255, 255, 255, 0.84);
        border-radius: 8px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.045);
    }
    .result-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 16px 0;
    }
    .result-metric {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        min-height: 88px;
    }
    .result-metric span {
        display: block;
        color: #64748B;
        font-size: 13px;
        margin-bottom: 6px;
    }
    .result-metric strong {
        color: #0F172A;
        display: block;
        font-size: 21px;
        line-height: 1.16;
        overflow-wrap: anywhere;
    }
    .risk-summary {
        border: 1px solid #DDE8EA;
        border-top: 5px solid var(--risk-color);
        background: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: center;
        margin-bottom: 14px;
    }
    .risk-summary span {
        color: #64748B;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .risk-summary h2 {
        color: var(--risk-color);
        font-size: 32px;
        line-height: 1.1;
        margin: 6px 0 8px 0;
        letter-spacing: 0;
    }
    .risk-summary p {
        color: #475569;
        margin: 0;
        font-size: 16px;
        line-height: 1.45;
    }
    .risk-summary p + p {
        margin-top: 6px;
        color: #64748B;
        font-size: 14px;
    }
    .risk-summary p strong {
        color: #102A43;
    }
    .probability-pill {
        flex: 0 0 auto;
        min-width: 112px;
        border-radius: 8px;
        padding: 13px 16px;
        background: color-mix(in srgb, var(--risk-color) 12%, white);
        color: var(--risk-color);
        text-align: center;
        font-size: 31px;
        font-weight: 820;
        border: 1px solid color-mix(in srgb, var(--risk-color) 24%, white);
    }
    .risk-scale {
        display: grid;
        grid-template-columns: 4fr 3fr 3fr;
        gap: 6px;
        margin: 0 0 18px 0;
    }
    .scale-segment {
        height: 10px;
        border-radius: 999px;
    }
    .scale-caption {
        display: grid;
        grid-template-columns: 4fr 3fr 3fr;
        gap: 6px;
        color: #64748B;
        font-size: 13px;
        margin: -12px 0 18px 0;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .insight-card {
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 16px;
        min-height: 150px;
    }
    .insight-card span {
        display: block;
        color: #0F766E;
        font-size: 13px;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 8px;
    }
    .insight-card p {
        margin: 0;
        color: #334155;
        font-size: 16px;
        line-height: 1.5;
    }
    .insight-card:last-child {
        grid-column: 1 / -1;
    }
    .factor-list {
        margin: 0;
        padding-left: 18px;
        color: #334155;
        font-size: 16px;
        line-height: 1.5;
    }
    .factor-list li + li {
        margin-top: 5px;
    }
    .explain-note {
        border: 1px solid #CFE7E3;
        background: #F7FEFB;
        border-radius: 8px;
        padding: 15px 16px;
        color: #24404A;
        margin: 10px 0 16px 0;
    }
    .explain-note strong {
        color: #0F766E;
    }
    .measurement-reference {
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 12px 14px;
        margin: -2px 0 14px 0;
    }
    .measurement-reference-head {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(112px, 0.9fr);
        gap: 14px;
        align-items: start;
        color: #102A43;
        font-size: 14px;
        line-height: 1.35;
    }
    .measurement-reference-head strong {
        color: #102A43;
        overflow-wrap: anywhere;
    }
    .measurement-reference-head span {
        color: #0F766E;
        font-weight: 700;
        text-align: right;
        overflow-wrap: anywhere;
    }
    .measurement-track {
        position: relative;
        height: 14px;
        margin: 38px 4px 40px 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, #DBEAFE 0% 7.7%, #DCFCE7 7.7% 30.8%, #FEF3C7 30.8% 38.5%, #FFEDD5 38.5% 46.2%, #FEE2E2 46.2% 100%);
    }
    .measurement-marker {
        position: absolute;
        top: -7px;
        width: 3px;
        height: 28px;
        background: #102A43;
        transform: translateX(-50%);
        border-radius: 2px;
    }
    .measurement-marker span {
        position: absolute;
        top: -25px;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        background: #FFFFFF;
        padding: 0 4px;
        border-radius: 4px;
        color: #102A43;
        font-size: 12px;
        font-weight: 700;
        z-index: 2;
    }
    .measurement-current {
        position: absolute;
        top: -5px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #0F766E;
        border: 3px solid #FFFFFF;
        box-shadow: 0 0 0 1px #0F766E;
        transform: translateX(-50%);
    }
    .measurement-current span {
        position: absolute;
        top: 26px;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        background: #FFFFFF;
        padding: 0 4px;
        border-radius: 4px;
        color: #0F766E;
        font-size: 12px;
        font-weight: 700;
        z-index: 2;
    }
    .measurement-reference-labels {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 4px;
        color: #64748B;
        font-size: 11px;
        line-height: 1.25;
    }
    .measurement-reference-labels span:nth-child(n+2) {
        text-align: center;
    }
    .measurement-reference-labels span:last-child {
        text-align: right;
    }
    .measurement-reference p {
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
        margin: 10px 0 0 0;
    }
    .conclusion-card {
        border: 1px solid #99D5CC;
        border-left: 5px solid #0F766E;
        border-radius: 8px;
        background: #F0FDFA;
        padding: 16px 18px;
        margin: 16px 0 8px 0;
        box-shadow: 0 8px 20px rgba(15, 118, 110, 0.07);
    }
    .conclusion-card h3 {
        color: #0F766E;
        font-size: 19px;
        margin: 0 0 7px 0;
    }
    .conclusion-card p {
        color: #134E4A;
        font-size: 16px;
        line-height: 1.55;
        margin: 0;
    }
    .recommendation-card {
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 16px 18px;
        margin: 12px 0 8px 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    .recommendation-card h3 {
        color: #102A43;
        font-size: 19px;
        margin: 0 0 8px 0;
    }
    .recommendation-list {
        margin: 0;
        padding-left: 20px;
        color: #334155;
        font-size: 15px;
        line-height: 1.55;
    }
    .recommendation-list li + li {
        margin-top: 6px;
    }
    .recommendation-list strong {
        color: #0F766E;
    }
    .urgent-note {
        border-left: 4px solid #DC2626;
        background: #FEF2F2;
        color: #991B1B;
        border-radius: 6px;
        padding: 11px 13px;
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.5;
    }
    .range-warning {
        border-left: 4px solid #D97706;
        background: #FFF7ED;
        color: #7C2D12;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 8px 0 16px 0;
        font-size: 15px;
        line-height: 1.5;
    }
    .range-warning strong {
        display: block;
        color: #9A3412;
        margin-bottom: 3px;
    }
    .range-warning p {
        margin: 0 0 4px 0;
    }
    .range-warning ul {
        margin: 4px 0 0 18px;
    }
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 18px 0;
    }
    .guide-card {
        background: #FFFFFF;
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        padding: 15px;
        min-height: 118px;
    }
    .guide-card strong {
        display: block;
        color: #102A43;
        font-size: 17px;
        margin-bottom: 6px;
    }
    .guide-card span {
        color: #475569;
        font-size: 15px;
        line-height: 1.5;
    }
    .table-wrap {
        overflow-x: auto;
        border: 1px solid #DDE8EA;
        border-radius: 8px;
        background: #FFFFFF;
        margin: 8px 0 18px 0;
    }
    .table-wrap table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
        table-layout: auto;
    }
    .table-wrap th {
        background: #EEF7F6;
        color: #102A43;
        text-align: left;
        padding: 12px 14px;
        border-bottom: 1px solid #DDE8EA;
        font-weight: 760;
        white-space: normal;
    }
    .table-wrap td {
        color: #334155;
        padding: 11px 14px;
        border-bottom: 1px solid #EEF2F7;
        vertical-align: top;
        min-width: 110px;
    }
    .table-wrap tr:last-child td {
        border-bottom: 0;
    }
    .table-wrap tbody tr:nth-child(even) {
        background: #FAFCFD;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #0F766E;
        background: #0F766E;
        color: #FFFFFF;
        font-weight: 700;
        min-height: 42px;
    }
    .stButton > button:hover {
        border-color: #115E59;
        background: #115E59;
        color: #FFFFFF;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stSlider"] label p,
    [data-testid="stSelectbox"] label p {
        color: #102A43 !important;
        font-weight: 650 !important;
        font-size: 15px !important;
    }
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #102A43 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] input {
        color: #102A43 !important;
    }
    [data-baseweb="select"] svg {
        fill: #0F766E !important;
    }
    button[data-baseweb="tab"] {
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0F766E !important;
        border-bottom-color: #0F766E !important;
    }
    button[data-baseweb="tab"] p {
        color: inherit !important;
        font-weight: 650 !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #64748B !important;
    }
    [data-testid="stMetricValue"] div {
        color: #0F172A !important;
    }
    @media (max-width: 900px) {
        .app-hero {
            display: block;
            padding: 18px;
        }
        .hero-stats {
            grid-template-columns: repeat(3, minmax(0, 1fr));
            min-width: unset;
            margin-top: 14px;
        }
        .stat {
            padding: 10px;
        }
        .stat strong {
            font-size: 18px;
        }
        .app-hero h1 {
            font-size: 28px;
        }
        .app-hero p {
            font-size: 15px;
        }
        .clinical-strip,
        .patient-grid,
        .guide-grid,
        .validation-summary {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .flag-grid {
            grid-template-columns: 1fr;
        }
        .insight-grid,
        .result-metrics {
            grid-template-columns: 1fr;
        }
        .insight-card:last-child {
            grid-column: auto;
        }
        .risk-summary {
            display: block;
        }
        .probability-pill {
            margin-top: 12px;
        }
    }
    @media (max-width: 520px) {
        .hero-stats,
        .clinical-strip,
        .patient-grid,
        .flag-grid,
        .result-metrics,
            .insight-grid,
        .guide-grid,
        .validation-summary {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 700px) {
        .patient-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


MODEL_NAME_VI = {
    "Logistic Regression": "Hồi quy Logistic",
    "K-Nearest Neighbors": "K láng giềng gần nhất",
    "Support Vector Machine": "Máy vector hỗ trợ (SVM)",
    "Random Forest": "Rừng ngẫu nhiên",
    "Gradient Boosting": "Tăng cường Gradient",
    "XGBoost": "XGBoost",
}

METRIC_LABELS_VI = {
    "model": "Mô hình",
    "selected_by_cv": "Được chọn bằng kiểm định chéo",
    "cv_recall": "Độ nhạy kiểm định chéo",
    "cv_f1_score": "Điểm F1 kiểm định chéo",
    "cv_roc_auc": "AUC ROC kiểm định chéo",
    "cv_selection_score": "Điểm chọn mô hình",
    "recall": "Độ nhạy trên tập kiểm tra",
    "roc_auc": "AUC ROC trên tập kiểm tra",
    "average_precision": "Độ chính xác trung bình",
}

THRESHOLD_LABELS_VI = {
    "threshold": "Ngưỡng",
    "precision": "Độ chính xác cảnh báo",
    "recall": "Độ nhạy",
    "specificity": "Độ đặc hiệu",
    "f2_score": "Điểm F2",
}

FEATURE_HELP = {
    "age": "Tuổi là yếu tố nền quan trọng vì nguy cơ tim mạch thường tăng theo tuổi.",
    "sex": "Giới tính được giữ theo mã thống nhất của các nguồn UCI để mô hình học khác biệt thống kê giữa hai nhóm.",
    "cp": "Hệ thống tự quy đổi từ các dấu hiệu người bệnh đã chọn; đây là nhóm tham chiếu của dữ liệu, không phải chẩn đoán.",
    "trestbps": "Đây là số tâm thu (số trên) của huyết áp lúc nghỉ. Mốc bình thường đầy đủ thường dưới 120/80 mmHg; bộ dữ liệu không có số tâm trương nên cần thận trọng khi diễn giải.",
    "chol": "Đây là cholesterol toàn phần; cần xem cùng LDL, HDL, triglyceride và các yếu tố khác, không kết luận chỉ từ số này.",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ. Hãy chọn đúng theo phiếu: bình thường, bất thường đoạn ST/sóng T hoặc dấu hiệu phì đại thất trái.",
    "thalach": "Nhịp tim cao nhất ghi được trong nghiệm pháp vận động; hãy lấy từ phiếu kết quả, không tự ước tính hoặc cố tập nặng để lấy số.",
    "exang": "Cho biết có hay không đau ngực khi gắng sức/vận động theo hỏi bệnh hoặc kết luận của nhân viên y tế.",
    "oldpeak": "Mức thay đổi của đoạn ST khi vận động so với lúc nghỉ. Đây là số liệu từ điện tâm đồ gắng sức, không phải điểm tự chấm tại nhà.",
}

CURRENT_SYMPTOM_HELP = (
    "Câu hỏi an toàn bổ sung, không được đưa vào mô hình: bệnh nhân có đang đau ngực, "
    "khó thở hoặc triệu chứng đáng lo ngay lúc này không?"
)

OBSERVED_CONTINUOUS_RANGES = {
    "age": (28.0, 77.0),
    "trestbps": (80.0, 200.0),
    "chol": (85.0, 603.0),
    "thalach": (60.0, 202.0),
    "oldpeak": (-2.6, 6.2),
}


CP_GUIDANCE = (
    "Hãy trả lời theo mô tả thực tế, không cần tự gọi tên bệnh: (1) khó chịu ở giữa hoặc sau xương ức, "
    "(2) xuất hiện khi gắng sức hoặc xúc động, (3) giảm khi nghỉ. "
    "Hệ thống chỉ dùng các câu trả lời này để tạo nhóm đầu vào cho mô hình."
)


def blood_pressure_assessment(systolic: float) -> tuple[str, str, str]:
    """Assess the systolic value only; the app does not collect diastolic pressure."""
    value = float(systolic)
    if value < 90:
        return (
            "Thấp theo mốc tâm thu",
            "#1D4ED8",
            "Nếu kèm chóng mặt, nhìn mờ, yếu hoặc ngất, cần được nhân viên y tế đánh giá. Hãy đo lại đủ cả tâm thu và tâm trương.",
        )
    if value < 120:
        return (
            "Trong vùng tâm thu tham khảo",
            "#15803D",
            "Mốc bình thường đầy đủ thường là dưới 120/80 mmHg; ứng dụng hiện chỉ có số tâm thu.",
        )
    if value < 130:
        return (
            "Cao hơn vùng tâm thu bình thường",
            "#A16207",
            "Nên đo lại đúng kỹ thuật vào các thời điểm khác nhau và theo dõi xu hướng, không kết luận từ một lần đo.",
        )
    if value < 140:
        return (
            "Tăng theo mốc tâm thu giai đoạn 1",
            "#C2410C",
            "Nên ghi nhận nhiều lần đo và trao đổi với nhân viên y tế để đánh giá cả tâm thu, tâm trương và yếu tố nguy cơ.",
        )
    if value < 180:
        return (
            "Tăng theo mốc tâm thu giai đoạn 2",
            "#B91C1C",
            "Nên sắp xếp đánh giá y tế; không tự thay đổi thuốc chỉ dựa trên thanh đo này.",
        )
    return (
        "Rất cao theo mốc tâm thu",
        "#991B1B",
        "Ngồi yên ít nhất 5 phút rồi đo lại đúng kỹ thuật và đo đủ hai số. Nếu vẫn từ 180 mmHg trở lên (hoặc số dưới từ 120 mmHg trở lên), cần liên hệ cơ sở y tế ngay trong ngày; nếu có đau ngực, khó thở, yếu liệt, nhìn mờ hay khó nói, cần gọi cấp cứu ngay.",
    )


def blood_pressure_reference_position(value: float, minimum: float = 80, maximum: float = 210) -> float:
    """Map the systolic value to the visual reference track."""
    bounded = min(max(float(value), minimum), maximum)
    return (bounded - minimum) / (maximum - minimum) * 100


def render_blood_pressure_reference(value: float) -> None:
    label, color, message = blood_pressure_assessment(value)
    current_position = blood_pressure_reference_position(value)
    normal_position = blood_pressure_reference_position(120)
    st.markdown(
        f"""
        <div class="measurement-reference">
            <div class="measurement-reference-head">
                <strong>Mốc tham khảo cho huyết áp tâm thu lúc nghỉ</strong>
                <span style="color:{color};">{label}</span>
            </div>
            <div class="measurement-track">
                <div class="measurement-marker" style="left:{normal_position:.2f}%;"><span>Mốc 120</span></div>
                <div class="measurement-current" style="left:{current_position:.2f}%;"><span>{format_decimal_vi(value, 0)} mmHg</span></div>
            </div>
            <div class="measurement-reference-labels">
                <span>&lt;90 thấp</span>
                <span>90–119</span>
                <span>120–129</span>
                <span>130–139</span>
                <span>≥140 cao</span>
            </div>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_value_meaning(feature: str, value: float) -> str:
    """Explain categorical values in language a non-specialist can read."""
    displayed = display_value(feature, value)
    continuous_labels = {
        "age": "Tuổi hiện tại",
        "trestbps": "Huyết áp tâm thu (số trên) lúc nghỉ",
        "chol": "Cholesterol toàn phần trong máu",
        "thalach": "Nhịp tim cao nhất khi vận động",
        "oldpeak": "Mức thay đổi đoạn ST khi vận động",
    }
    if feature in continuous_labels:
        if feature == "oldpeak":
            displayed = format_decimal_vi(value, 1)
        return f"{continuous_labels[feature]} là {displayed}"
    if feature == "slope":
        meanings = {
            1.0: "Đoạn ST hướng lên khi vận động",
            2.0: "Đoạn ST gần như nằm ngang khi vận động",
            3.0: "Đoạn ST hướng xuống khi vận động",
        }
        return meanings.get(float(value), f"Giá trị ST: {displayed}")
    if feature == "cp":
        meanings = {
            1.0: "đau ngực phù hợp cả 3 dấu hiệu điển hình",
            2.0: "đau ngực phù hợp 2 trong 3 dấu hiệu điển hình",
            3.0: "đau ngực phù hợp 0–1 trong 3 dấu hiệu điển hình",
            4.0: "chưa ghi nhận triệu chứng đau ngực rõ",
        }
        return meanings.get(float(value), displayed)
    if feature == "exang":
        return "có đau ngực khi gắng sức" if float(value) == 1.0 else "không ghi nhận đau ngực khi gắng sức"
    if feature == "fbs":
        return "có đường huyết lúc đói trên 120 mg/dL theo mã của bộ dữ liệu" if float(value) == 1.0 else "không vượt 120 mg/dL theo mã của bộ dữ liệu"
    if feature == "restecg":
        meanings = {
            0.0: "điện tâm đồ lúc nghỉ được ghi là bình thường",
            1.0: "điện tâm đồ lúc nghỉ có bất thường ở đoạn ST hoặc sóng T",
            2.0: "điện tâm đồ lúc nghỉ có dấu hiệu phì đại thất trái",
        }
        return meanings.get(float(value), displayed)
    if feature == "ca":
        return f"có {displayed} mạch máu chính được ghi nhận trên chụp mạch"
    if feature == "thal":
        meanings = {
            3.0: "kết quả tưới máu cơ tim được ghi là bình thường",
            6.0: "kết quả cho thấy khiếm khuyết cố định theo mã dữ liệu",
            7.0: "kết quả cho thấy khiếm khuyết có thể hồi phục theo mã dữ liệu",
        }
        return meanings.get(float(value), displayed)
    if feature == "sex":
        return f"giá trị giới tính trong hồ sơ là {displayed}"
    return ""


def model_name_vi(name: str) -> str:
    if name.startswith("Calibrated "):
        raw_name = name.replace("Calibrated ", "", 1)
        return f"{MODEL_NAME_VI.get(raw_name, raw_name)} đã hiệu chỉnh xác suất"
    return MODEL_NAME_VI.get(name, name)


def format_decimal_vi(value: float, digits: int = 2) -> str:
    """Format a decimal with Vietnamese decimal punctuation."""
    return f"{float(value):.{digits}f}".replace(".", ",")


def format_percent_vi(value: float, digits: int = 1, *, sign: bool = False) -> str:
    """Format a probability or probability delta as a Vietnamese percentage."""
    numeric = float(value) * 100
    if abs(numeric) < 0.05:
        numeric = 0.0
    prefix = "+" if sign and numeric > 0 else ""
    return f"{prefix}{numeric:.{digits}f}".replace(".", ",") + "%"


def format_percentage_points_vi(value: float, digits: int = 1, *, sign: bool = False) -> str:
    """Format a probability difference as Vietnamese percentage points."""
    numeric = float(value) * 100
    if abs(numeric) < 0.05:
        numeric = 0.0
    prefix = "+" if sign and numeric > 0 else ""
    return f"{prefix}{numeric:.{digits}f}".replace(".", ",") + " điểm phần trăm"


def format_delta_vi(value: float) -> str:
    """Show a probability difference as percentage points, not relative percent."""
    return format_percentage_points_vi(value, sign=True)


def impact_direction(delta: float) -> str:
    if abs(float(delta)) < 0.0005:
        return "Không đáng kể"
    return "Tăng xác suất nhóm có bệnh theo mô hình" if delta > 0 else "Giảm xác suất nhóm có bệnh theo mô hình"


def impact_explanation(
    delta: float,
    feature: str | None = None,
    patient_value: float | None = None,
    reference_value: float | None = None,
) -> str:
    delta = float(delta)
    meaning = feature_value_meaning(feature, patient_value) if feature and patient_value is not None else ""
    meaning_prefix = f"Giá trị hiện tại: {meaning}. " if meaning else ""
    if feature and reference_value is not None:
        reference_meaning = feature_value_meaning(feature, reference_value)
        if reference_meaning:
            meaning_prefix += f"Giá trị tham chiếu: {reference_meaning}. "
    if abs(delta) < 0.0005:
        return f"{meaning_prefix}Không làm thay đổi đáng kể xác suất dự đoán so với giá trị tham chiếu."
    amount = format_percentage_points_vi(abs(delta))
    verb = "tăng" if delta > 0 else "giảm"
    return f"{meaning_prefix}Làm {verb} xác suất dự đoán khoảng {amount} so với giá trị tham chiếu."


def format_metric(value: float) -> str:
    return format_decimal_vi(value, 3)


def yes_no_vi(value) -> str:
    if isinstance(value, bool):
        return "Có" if value else "Không"
    return "Có" if str(value).strip().lower() == "true" else "Không"


def render_explain_note(title: str, body: str) -> None:
    st.markdown(
        f'<div class="explain-note"><strong>{title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )


def render_table(frame: pd.DataFrame) -> None:
    table_html = frame.to_html(index=False, escape=False)
    st.markdown(f'<div class="table-wrap">{table_html}</div>', unsafe_allow_html=True)


def render_input_group(title: str, body: str) -> None:
    st.markdown(
        f'<div class="input-group-title"><strong>{title}</strong><span>{body}</span></div>',
        unsafe_allow_html=True,
    )


def patient_flags(values: dict[str, float]) -> list[tuple[str, str, str]]:
    """Create short, plain-language checks with a visual urgency level."""
    flags: list[tuple[str, str, str]] = []
    if has_current_warning(values):
        flags.append(
            (
                "Triệu chứng đang xảy ra",
                "Đang có đau ngực, khó thở hoặc triệu chứng đáng lo. Hãy dừng vận động và liên hệ cấp cứu; không chờ kết quả ứng dụng.",
                "critical",
            )
        )
    if has_unclear_symptoms(values):
        flags.append(
            (
                "Chưa rõ kiểu đau ngực",
                "Chưa đủ thông tin để nhận biết kiểu đau ngực. Hãy hỏi lại theo phiếu bệnh; không nên tự đoán.",
                "warning",
            )
        )
    if values.get("symptom_radiation", 0.0) == 1.0:
        flags.append(
            (
                "Đau lan sang vùng khác",
                "Cơn đau lan ra tay, vai, hàm hoặc lưng. Đây là thông tin cần nói rõ khi trao đổi với nhân viên y tế.",
                "warning",
            )
        )
    associated_symptoms = []
    associated_labels = {
        "symptom_dyspnea": "khó thở",
        "symptom_cold_sweat": "vã mồ hôi lạnh",
        "symptom_nausea": "buồn nôn hoặc nôn",
        "symptom_dizziness_syncope": "chóng mặt, choáng hoặc ngất",
        "symptom_palpitations": "hồi hộp hoặc tim đập bất thường",
    }
    for key, label in associated_labels.items():
        if values.get(key, 0.0) == 1.0:
            associated_symptoms.append(label)
    if associated_symptoms:
        symptom_text = ", ".join(associated_symptoms)
        severity = "critical" if has_current_warning(values) and any(
            key in associated_labels and values.get(key, 0.0) == 1.0
            for key in ("symptom_dyspnea", "symptom_cold_sweat", "symptom_dizziness_syncope")
        ) else "warning"
        flags.append(
            (
                "Triệu chứng đi kèm",
                f"Có ghi nhận {symptom_text}. Hãy nói rõ thời điểm xuất hiện; nếu đang xảy ra hoặc nặng lên, cần được đánh giá ngay.",
                severity,
            )
        )
    fluid_related_symptoms = []
    fluid_related_labels = {
        "symptom_orthopnea": "khó thở khi nằm",
        "symptom_nocturnal_dyspnea": "thức giấc ban đêm vì khó thở",
        "symptom_leg_swelling": "phù chân hoặc mắt cá",
    }
    for key, label in fluid_related_labels.items():
        if values.get(key, 0.0) == 1.0:
            fluid_related_symptoms.append(label)
    if fluid_related_symptoms:
        flags.append(
            (
                "Triệu chứng cần đánh giá thêm",
                "Có ghi nhận "
                + ", ".join(fluid_related_symptoms)
                + ". Nên mang thông tin này khi đi khám; không tự kết luận nguyên nhân.",
                "warning",
            )
        )
    if values.get("symptom_duration", 0.0) >= 3.0:
        flags.append(
            (
                "Cơn đau kéo dài",
                "Cơn đau kéo dài trên 20 phút hoặc không giảm. Nếu đang xảy ra, hãy dừng vận động và liên hệ cơ sở y tế ngay.",
                "critical" if has_current_warning(values) else "warning",
            )
        )
    if values.get("symptom_severity", 0.0) >= 7.0:
        flags.append(
            (
                "Mức độ khó chịu cao",
                "Mức độ đau hoặc khó chịu được đánh giá từ 7/10 trở lên. Không nên cố chịu đựng hoặc tự tập nặng để kiểm tra.",
                "critical" if has_current_warning(values) else "warning",
            )
        )
    if values["trestbps"] < 90:
        flags.append(
            (
                "Huyết áp tâm thu thấp",
                "Số trên dưới 90 mmHg. Hãy đo lại đúng cách; nếu chóng mặt, mệt lả hoặc ngất, nên được đánh giá y tế.",
                "warning",
            )
        )
    elif values["trestbps"] >= 180:
        flags.append(
            (
                "Huyết áp tâm thu rất cao",
                "Số trên từ 180 mmHg. Ngồi nghỉ và đo lại đủ hai số; nếu kèm đau ngực, khó thở, yếu liệt hoặc nhìn mờ, hãy gọi cấp cứu.",
                "critical",
            )
        )
    elif values["trestbps"] >= 140:
        flags.append(
            (
                "Huyết áp tâm thu cao",
                "Số trên từ 140 mmHg. Nên ghi lại nhiều lần đo và trao đổi với nhân viên y tế; không kết luận chỉ từ một lần đo.",
                "warning",
            )
        )
    elif values["trestbps"] >= 130:
        flags.append(
            (
                "Huyết áp tâm thu hơi cao",
                "Số trên nằm trong khoảng 130–139 mmHg. Nên theo dõi nhiều lần và xem cùng số dưới.",
                "info",
            )
        )

    diastolic = values.get("trestbps_diastolic")
    if diastolic is not None and float(diastolic) >= 120:
        flags.append(
            (
                "Huyết áp tâm trương rất cao",
                "Số dưới từ 120 mmHg. Hãy đo lại đủ hai số; nếu có triệu chứng đáng lo, cần liên hệ cấp cứu.",
                "critical",
            )
        )
    elif diastolic is not None and float(diastolic) >= 90:
        flags.append(
            (
                "Huyết áp tâm trương cao",
                "Số dưới từ 90 mmHg. Nên ghi lại nhiều lần đo và trao đổi với nhân viên y tế.",
                "warning",
            )
        )

    if values["chol"] >= 240:
        flags.append(
            (
                "Cholesterol toàn phần cao",
                "Cholesterol toàn phần từ 240 mg/dL trở lên là cao theo mốc tham khảo. Nên xem thêm LDL, HDL và triglyceride.",
                "warning",
            )
        )
    elif values["chol"] >= 200:
        flags.append(
            (
                "Cholesterol toàn phần cần theo dõi",
                "Cholesterol toàn phần từ 200 mg/dL trở lên. Nên xem cùng LDL, HDL và triglyceride để đánh giá đầy đủ hơn.",
                "info",
            )
        )

    if values["exang"] == 1.0:
        flags.append(
            (
                "Đau ngực khi vận động",
                "Có ghi nhận đau hoặc khó chịu ở ngực khi gắng sức. Nên mang kết quả đi khám và không cố tập nặng để tự kiểm tra.",
                "warning",
            )
        )
    if values["oldpeak"] >= 2.0:
        flags.append(
            (
                "Thay đổi đoạn ST khi gắng sức",
                "Kết quả nghiệm pháp gắng sức cho thấy đoạn ST thay đổi khá rõ. Cần xem cùng triệu chứng và các kết quả khác.",
                "warning",
            )
        )
    elif values["oldpeak"] > 0:
        flags.append(
            (
                "Có thay đổi đoạn ST khi gắng sức",
                "Kết quả nghiệm pháp gắng sức cho thấy đoạn ST có thay đổi. Nên để nhân viên y tế đọc cùng các chỉ số khác.",
                "info",
            )
        )
    risk_labels = {
        "history_hypertension": "tiền sử tăng huyết áp",
        "history_diabetes": "tiền sử tiểu đường",
        "history_dyslipidemia": "tiền sử mỡ máu cao",
        "history_cardiovascular": "tiền sử bệnh tim hoặc đột quỵ",
        "family_history_early_cvd": "tiền sử gia đình đáng chú ý",
    }
    reported_risks = [
        label for key, label in risk_labels.items() if values.get(key, 0.0) == 1.0
    ]
    if values.get("smoking_status", 0.0) in (1.0, 2.0):
        reported_risks.append("tiền sử hút thuốc")
    if reported_risks:
        flags.append(
            (
                "Yếu tố nguy cơ đã khai báo",
                "Có "
                + ", ".join(reported_risks)
                + ". Các thông tin này bổ sung bối cảnh, chưa được đưa vào xác suất mô hình hiện tại.",
                "info",
            )
        )

    if not flags:
        flags.append(
            (
                "Tổng quan",
                "Chưa thấy tín hiệu vượt các mốc kiểm tra nhanh. Điều này không loại trừ bệnh; vẫn cần xem cùng triệu chứng và ý kiến nhân viên y tế.",
                "info",
            )
        )
    return flags[:4]


def render_patient_flags(values: dict[str, float]) -> None:
    cards = ['<div class="flag-grid">']
    for title, body, severity in patient_flags(values):
        klass = f"flag-card {severity}"
        cards.append(f'<div class="{klass}"><strong>{title}</strong>{body}</div>')
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def load_threshold_analysis() -> pd.DataFrame:
    if not THRESHOLD_ANALYSIS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(THRESHOLD_ANALYSIS_PATH)


def option_select(feature: str, *, key: str) -> float:
    info = FEATURE_INFO[feature]
    options = {float(raw): label for raw, label in info["options"].items()}
    values = list(options.keys())
    default = float(info["default"])
    index = values.index(default) if default in values else 0
    return float(
        st.selectbox(
            info["ui_label"],
            options=values,
            index=index,
            format_func=lambda value: options[float(value)],
            key=key,
            help=FEATURE_HELP.get(feature, info.get("description_vi", "")),
        )
    )


def number_input(feature: str, *, key: str) -> float:
    info = FEATURE_INFO[feature]
    value = st.slider(
        info["ui_label"],
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=float(info["default"]),
        step=float(info["step"]),
        key=key,
        help=FEATURE_HELP.get(feature, info.get("description_vi", "")),
    )
    if feature == "trestbps":
        render_blood_pressure_reference(float(value))
    return float(value)


def current_symptom_label(value: float) -> str:
    return "Có triệu chứng đáng lo lúc này" if float(value) == 1.0 else "Chưa ghi nhận triệu chứng đáng lo lúc này"


def collect_current_warning_inputs() -> dict[str, float]:
    """Collect independent red-flag answers before the model result is shown."""
    st.caption(
        "Nếu có dấu hiệu đang xảy ra ngay lúc này, hãy dừng vận động và liên hệ cấp cứu; "
        "không chờ kết quả của ứng dụng."
    )
    current_left, current_right = st.columns(2)
    with current_left:
        chest_pain = st.checkbox(
            "Đang đau hoặc tức ngực ngay lúc này",
            key="current_chest_pain",
        )
        severe_dyspnea = st.checkbox(
            "Đang khó thở nhiều ngay lúc này",
            key="current_severe_dyspnea",
        )
        syncope = st.checkbox(
            "Đang choáng nhiều hoặc vừa ngất",
            key="current_syncope",
        )
    with current_right:
        cold_sweat = st.checkbox(
            "Đang vã mồ hôi lạnh",
            key="current_cold_sweat",
        )
        radiating_pain = st.checkbox(
            "Cơn đau đang lan tới tay, vai, lưng, cổ hoặc hàm",
            key="current_radiating_pain",
        )
    current_values = {
        "current_chest_pain": float(chest_pain),
        "current_severe_dyspnea": float(severe_dyspnea),
        "current_syncope": float(syncope),
        "current_cold_sweat": float(cold_sweat),
        "current_radiating_pain": float(radiating_pain),
    }
    current_values["current_symptoms"] = float(any(current_values.values()))
    return current_values


def collect_chest_symptom_inputs() -> tuple[float, float, float]:
    """Collect plain-language symptom answers and map them to model fields."""
    status = st.radio(
        "Trong đợt triệu chứng đang khai thác, bệnh nhân có đau hoặc khó chịu ở ngực không?",
        options=["Có", "Không", "Chưa rõ"],
        index=1,
        horizontal=True,
        key="chest_pain_status",
        help="Nếu chưa có kết luận rõ, chọn Chưa rõ; không cần tự gọi tên loại đau ngực.",
    )
    if status == "Có":
        st.caption("Chọn những mô tả đúng với cơn đau; có thể chọn nhiều mục.")
        criterion_left, criterion_right = st.columns(2)
        with criterion_left:
            behind_sternum = st.checkbox(
                "Khó chịu ở giữa hoặc sau xương ức",
                key="chest_pain_behind_sternum",
            )
            with_exertion = st.checkbox(
                "Xuất hiện khi gắng sức hoặc xúc động",
                key="chest_pain_with_exertion",
            )
        with criterion_right:
            relieved_by_rest = st.checkbox(
                "Giảm khi nghỉ",
                key="chest_pain_relieved_by_rest",
            )
        criterion_count = sum([behind_sternum, with_exertion, relieved_by_rest])
        if criterion_count == 3:
            cp_value = 1.0
            summary = "Có đủ 3 đặc điểm tham khảo"
        elif criterion_count == 2:
            cp_value = 2.0
            summary = "Có 2/3 đặc điểm tham khảo"
        else:
            cp_value = 3.0
            summary = f"Có {criterion_count}/3 đặc điểm tham khảo"
        st.info(
            f"Hệ thống tạm xếp: {summary}. Đây chỉ là cách mã hóa đầu vào cho mô hình, không phải chẩn đoán.",
        )
        return cp_value, float(with_exertion), 0.0
    if status == "Chưa rõ":
        st.warning(
            "Chưa đủ thông tin để phân loại đặc điểm đau ngực. Không nên tự suy đoán; hãy nhập theo phiếu hỏi bệnh hoặc trao đổi với nhân viên y tế.",
        )
        return 4.0, 0.0, 1.0
    return 4.0, 0.0, 0.0


def collect_additional_symptom_inputs() -> dict[str, float]:
    """Collect standard associated symptoms for safety context and record keeping."""
    st.caption(
        "Các câu hỏi dưới đây bổ sung cho hỏi bệnh. Chúng giúp nhận diện dấu hiệu cần chú ý và được lưu cùng hồ sơ; mô hình hiện tại chưa được huấn luyện trực tiếp với các biến này."
    )
    radiation_options = {"Không": 0.0, "Có": 1.0, "Chưa rõ": 2.0}
    duration_options = {
        "Chưa rõ hoặc không nhớ": 0.0,
        "Dưới 5 phút": 1.0,
        "5–20 phút": 2.0,
        "Trên 20 phút": 3.0,
        "Kéo dài liên tục hoặc không giảm": 4.0,
    }
    onset_options = {
        "Chưa rõ hoặc không nhớ": 0.0,
        "Trong 1 giờ qua": 1.0,
        "Trong 24 giờ qua": 2.0,
        "Từ 1–7 ngày": 3.0,
        "Trên 7 ngày hoặc tái diễn nhiều lần": 4.0,
    }
    binary_options = {"Không": 0.0, "Có": 1.0, "Chưa rõ": 2.0}
    symptom_left, symptom_right = st.columns(2)
    with symptom_left:
        radiation = st.selectbox(
            "Cơn đau có lan ra tay, vai, hàm hoặc lưng không?",
            options=list(radiation_options),
            index=0,
            key="symptom_radiation_label",
        )
        duration = st.selectbox(
            "Cơn đau hoặc khó chịu kéo dài bao lâu?",
            options=list(duration_options),
            index=0,
            key="symptom_duration_label",
        )
        onset = st.selectbox(
            "Triệu chứng bắt đầu từ khi nào?",
            options=list(onset_options),
            index=0,
            key="symptom_onset_label",
        )
        at_rest = st.selectbox(
            "Cơn đau có xuất hiện khi nghỉ không?",
            options=list(binary_options),
            index=0,
            key="symptom_at_rest_label",
        )
        worsening = st.selectbox(
            "Triệu chứng có tăng dần hoặc tái diễn thường xuyên hơn không?",
            options=list(binary_options),
            index=0,
            key="symptom_worsening_label",
        )
        severity = st.slider(
            "Mức độ đau hoặc khó chịu cao nhất (0–10)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="symptom_severity",
            help="Chọn 0 nếu không đau hoặc chưa rõ mức độ.",
        )
    with symptom_right:
        st.markdown("**Triệu chứng đi kèm**")
        dyspnea = st.checkbox("Khó thở", key="symptom_dyspnea")
        cold_sweat = st.checkbox("Vã mồ hôi lạnh", key="symptom_cold_sweat")
        nausea = st.checkbox("Buồn nôn hoặc nôn", key="symptom_nausea")
        dizziness_syncope = st.checkbox(
            "Chóng mặt, choáng hoặc ngất",
            key="symptom_dizziness_syncope",
        )
        palpitations = st.checkbox(
            "Hồi hộp hoặc tim đập bất thường",
            key="symptom_palpitations",
        )
        orthopnea = st.checkbox(
            "Khó thở khi nằm",
            key="symptom_orthopnea",
        )
        nocturnal_dyspnea = st.checkbox(
            "Thức giấc ban đêm vì khó thở",
            key="symptom_nocturnal_dyspnea",
        )
        leg_swelling = st.checkbox(
            "Phù chân hoặc mắt cá chân",
            key="symptom_leg_swelling",
        )
    return {
        "symptom_radiation": radiation_options[radiation],
        "symptom_duration": duration_options[duration],
        "symptom_onset": onset_options[onset],
        "symptom_at_rest": binary_options[at_rest],
        "symptom_worsening": binary_options[worsening],
        "symptom_severity": float(severity),
        "symptom_dyspnea": float(dyspnea),
        "symptom_cold_sweat": float(cold_sweat),
        "symptom_nausea": float(nausea),
        "symptom_dizziness_syncope": float(dizziness_syncope),
        "symptom_palpitations": float(palpitations),
        "symptom_orthopnea": float(orthopnea),
        "symptom_nocturnal_dyspnea": float(nocturnal_dyspnea),
        "symptom_leg_swelling": float(leg_swelling),
    }


def collect_risk_history_inputs() -> dict[str, float | None]:
    """Collect risk factors for safety context; these are not model features yet."""
    st.caption(
        "Các yếu tố dưới đây giúp bổ sung đánh giá nguy cơ. Chúng chưa làm thay đổi xác suất mô hình "
        "cho đến khi có dữ liệu huấn luyện tương ứng."
    )
    history_options = {"Không": 0.0, "Có": 1.0, "Chưa rõ": 2.0}
    risk_left, risk_right = st.columns(2)
    with risk_left:
        hypertension = st.selectbox(
            "Đã từng được chẩn đoán tăng huyết áp chưa?",
            options=list(history_options),
            index=0,
            key="history_hypertension_label",
        )
        diabetes = st.selectbox(
            "Đã từng được chẩn đoán tiểu đường chưa?",
            options=list(history_options),
            index=0,
            key="history_diabetes_label",
        )
        dyslipidemia = st.selectbox(
            "Đã từng được chẩn đoán mỡ máu cao chưa?",
            options=list(history_options),
            index=0,
            key="history_dyslipidemia_label",
        )
        cardiovascular = st.selectbox(
            "Từng mắc bệnh tim, đột quỵ hoặc rối loạn nhịp chưa?",
            options=list(history_options),
            index=0,
            key="history_cardiovascular_label",
        )
    with risk_right:
        family_history = st.selectbox(
            "Gia đình có người mắc bệnh tim sớm hoặc đột tử không?",
            options=list(history_options),
            index=0,
            key="family_history_early_cvd_label",
        )
        smoking_options = {
            "Chưa từng hút": 0.0,
            "Đã từng hút, hiện đã bỏ": 1.0,
            "Đang hút": 2.0,
            "Chưa rõ": 3.0,
        }
        smoking = st.selectbox(
            "Tình trạng hút thuốc lá hiện nay",
            options=list(smoking_options),
            index=0,
            key="smoking_status_label",
        )
        has_resting_hr = st.checkbox(
            "Có số nhịp tim lúc nghỉ đã đo",
            value=False,
            key="has_resting_hr",
        )
        resting_hr: float | None = None
        if has_resting_hr:
            resting_hr = float(
                st.number_input(
                    "Nhịp tim lúc nghỉ (lần/phút)",
                    min_value=30,
                    max_value=220,
                    value=70,
                    step=1,
                    key="resting_hr",
                )
            )
    return {
        "history_hypertension": history_options[hypertension],
        "history_diabetes": history_options[diabetes],
        "history_dyslipidemia": history_options[dyslipidemia],
        "history_cardiovascular": history_options[cardiovascular],
        "family_history_early_cvd": history_options[family_history],
        "smoking_status": smoking_options[smoking],
        "resting_hr": resting_hr,
    }


def render_dataset_range_warning(values: dict[str, float]) -> None:
    out_of_range = []
    for feature, (minimum, maximum) in OBSERVED_CONTINUOUS_RANGES.items():
        value = float(values[feature])
        if value < minimum or value > maximum:
            out_of_range.append(
                f"{FEATURE_INFO[feature]['ui_label']}: {format_decimal_vi(value, 1)} "
                f"(dữ liệu học: {format_decimal_vi(minimum, 1)}–{format_decimal_vi(maximum, 1)})"
            )
    if out_of_range:
        items = "".join(f"<li>{item}</li>" for item in out_of_range)
        st.markdown(
            f"""
            <div class="range-warning">
                <strong>Giá trị nằm ngoài khoảng dữ liệu học</strong>
            <p>Mô hình vẫn có thể trả kết quả, nhưng độ tin cậy có thể thấp hơn vì hồ sơ này nằm ngoài khoảng quan sát của bộ dữ liệu UCI đã mở rộng.</p>
                <ul>{items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def collect_patient_inputs() -> tuple[str, dict[str, float]]:
    st.markdown('<p class="section-title">Thông tin bệnh nhân</p>', unsafe_allow_html=True)
    patient_name = st.text_input(
        "Tên bệnh nhân",
        placeholder="Nhập tên để nhận diện hồ sơ khi lưu",
        max_chars=120,
        key="patient_name",
    ).strip()
    st.caption("Tên chỉ được dùng để nhận diện hồ sơ trong lịch sử lưu cục bộ của ứng dụng.")
    render_explain_note(
        "Nhập theo hồ sơ lâm sàng",
        f"Có {len(FEATURES)} trường được đưa vào mô hình. Các dấu hiệu cấp cứu được hỏi riêng để cảnh báo an toàn và không làm thay đổi xác suất. Không tự đoán chỉ số xét nghiệm; nếu chưa có kết quả, hãy trao đổi với nhân viên y tế.",
    )
    values: dict[str, float] = {}

    render_input_group(
        "1. Thông tin cơ bản",
        "Các thông tin nền giúp mô hình đặt hồ sơ bệnh nhân vào đúng nhóm tham chiếu.",
    )
    base_left, base_right = st.columns(2)
    with base_left:
        values["age"] = number_input("age", key="age")
    with base_right:
        values["sex"] = option_select("sex", key="sex")

    render_input_group(
        "2. Kiểm tra nhanh dấu hiệu hiện tại",
        "Nếu có dấu hiệu đang xảy ra ngay lúc này, hãy ưu tiên xử trí y tế và không chờ kết quả của ứng dụng.",
    )
    values.update(collect_current_warning_inputs())

    render_input_group(
        "3. Triệu chứng khi hỏi bệnh",
        "Chỉ các câu hỏi có biến tương ứng trong dữ liệu huấn luyện được dùng để dự đoán: kiểu đau ngực và đau ngực khi gắng sức.",
    )
    values["cp"], values["exang"], values["symptom_info_unclear"] = collect_chest_symptom_inputs()
    render_explain_note("Cách chọn kiểu đau ngực", CP_GUIDANCE)

    with st.expander("4. Chỉ số đo và xét nghiệm — chỉ nhập khi có kết quả", expanded=False):
        render_input_group(
            "Chỉ số đo và xét nghiệm",
            "Nhập theo số đo hoặc phiếu xét nghiệm thật; không tự ước tính nếu chưa có dữ liệu.",
        )
        measure_left, measure_right = st.columns(2)
        with measure_left:
            values["trestbps"] = number_input("trestbps", key="trestbps")
            has_diastolic = st.checkbox(
                "Có số dưới của huyết áp trong hồ sơ",
                value=False,
                key="has_trestbps_diastolic",
                help="Số dưới chỉ dùng để bổ sung diễn giải; chưa đưa vào mô hình hiện tại.",
            )
            if has_diastolic:
                values["trestbps_diastolic"] = float(
                    st.number_input(
                        "Huyết áp tâm trương (số dưới, mmHg)",
                        min_value=30,
                        max_value=150,
                        value=80,
                        step=1,
                        key="trestbps_diastolic",
                    )
                )
                if values["trestbps_diastolic"] >= 120:
                    st.warning(
                        "Số dưới đang rất cao; đo lại đủ hai số và liên hệ cơ sở y tế. Nếu có triệu chứng cảnh báo, gọi cấp cứu.",
                    )
                elif values["trestbps_diastolic"] >= 90:
                    st.caption("Số dưới đang cao theo mốc tham khảo; nên ghi lại nhiều lần đo và trao đổi với nhân viên y tế.")
            else:
                values["trestbps_diastolic"] = None
                st.caption("Chưa nhập số dưới — kết quả huyết áp chỉ được diễn giải theo số tâm thu.")
            values["chol"] = number_input("chol", key="chol")
            st.caption("Đây là cholesterol toàn phần; nên xem thêm LDL, HDL và triglyceride.")
        with measure_right:
            values["thalach"] = number_input("thalach", key="thalach")
            st.caption("Chỉ nhập số lấy từ phiếu nghiệm pháp vận động; không tự ước tính.")

    with st.expander("5. Kết quả thăm dò tim mạch — chỉ nhập theo phiếu", expanded=False):
        render_input_group(
            "Kết quả thăm dò tim mạch",
            "Nhóm này lấy từ điện tâm đồ, nghiệm pháp gắng sức hoặc kết quả thăm dò mạch.",
        )
        cardio_left, cardio_right = st.columns(2)
        with cardio_left:
            values["restecg"] = option_select("restecg", key="restecg")
        with cardio_right:
            values["oldpeak"] = number_input("oldpeak", key="oldpeak")

    render_dataset_range_warning(values)
    return patient_name, values


def clinical_interpretation(probability: float) -> str:
    if probability < 0.4:
        return (
            "Với những thông tin đã nhập, hệ thống chưa thấy nhiều tín hiệu giống nhóm có bệnh trong dữ liệu học. "
            "Điều này không khẳng định bệnh nhân hoàn toàn không có bệnh và vẫn cần xem cùng triệu chứng thực tế."
        )
    if probability < 0.7:
        return (
            "Kết quả đang ở vùng cần chú ý: một số thông tin đã nhập khiến hệ thống dự đoán gần với nhóm có bệnh hơn. "
            "Điều này không có nghĩa bệnh nhân chắc chắn mắc bệnh, nhưng nên kiểm tra lại và trao đổi với nhân viên y tế."
        )
    return (
        "Kết quả đang ở vùng cao: hồ sơ có nhiều thông tin khiến hệ thống dự đoán gần với nhóm có bệnh hơn. "
        "Bệnh nhân nên được nhân viên y tế đánh giá sớm; đây vẫn không phải là chẩn đoán xác định."
    )


def next_steps(probability: float) -> str:
    if probability < 0.4:
        return (
            "Gợi ý tham khảo: duy trì lối sống lành mạnh, theo dõi sức khỏe định kỳ và cập nhật dữ liệu nếu "
            "xuất hiện triệu chứng mới."
        )
    if probability < 0.7:
        return (
            "Gợi ý tham khảo: kiểm tra lại các chỉ số đã nhập và ghi nhận triệu chứng khi vận động. Nếu đau ngực, "
            "khó thở hoặc mệt bất thường, nên liên hệ cơ sở y tế để được tư vấn."
        )
    return (
        "Gợi ý tham khảo: sắp xếp đánh giá y tế, mang theo các chỉ số đã nhập và kết quả xét nghiệm gần nhất. "
        "Nếu có đau ngực dữ dội hoặc khó thở cấp, cần liên hệ cơ sở y tế ngay."
    )


def has_chest_symptoms(values: dict[str, float]) -> bool:
    return float(values["cp"]) != 4.0 or float(values["exang"]) == 1.0


def has_current_warning(values: dict[str, float]) -> bool:
    current_keys = (
        "current_chest_pain",
        "current_severe_dyspnea",
        "current_syncope",
        "current_cold_sweat",
        "current_radiating_pain",
    )
    return float(values.get("current_symptoms", 0.0)) == 1.0 or any(
        float(values.get(key, 0.0) or 0.0) == 1.0 for key in current_keys
    )


def supplemental_assessment(values: dict[str, float]) -> tuple[str, str, str]:
    """Summarize the non-model questions without inventing a probability."""
    if has_current_warning(values):
        return (
            "Cảnh báo cần xử trí ngay",
            "Có dấu hiệu đang xảy ra có thể cần đánh giá cấp cứu. Dừng vận động và liên hệ cấp cứu; không chờ xác suất mô hình.",
            "critical",
        )

    symptom_keys = (
        "symptom_orthopnea",
        "symptom_nocturnal_dyspnea",
        "symptom_leg_swelling",
        "symptom_palpitations",
        "symptom_dyspnea",
        "symptom_dizziness_syncope",
    )
    has_concerning_symptom = any(
        float(values.get(key, 0.0) or 0.0) == 1.0 for key in symptom_keys
    )
    risk_keys = (
        "history_hypertension",
        "history_diabetes",
        "history_dyslipidemia",
        "history_cardiovascular",
        "family_history_early_cvd",
    )
    risk_count = sum(
        float(values.get(key, 0.0) or 0.0) == 1.0 for key in risk_keys
    )
    if values.get("smoking_status", 0.0) in (1.0, 2.0):
        risk_count += 1

    if has_concerning_symptom or risk_count >= 2:
        return (
            "Nên khám và đánh giá thêm",
            "Các câu hỏi bổ sung ghi nhận triệu chứng hoặc yếu tố nguy cơ cần được trao đổi với nhân viên y tế. Thông tin này chưa được cộng vào xác suất mô hình.",
            "warning",
        )
    return (
        "Chưa ghi nhận dấu hiệu cấp cứu hiện tại",
        "Phần kiểm tra nhanh chưa ghi nhận dấu hiệu cần xử trí ngay. Kết quả mô hình vẫn không loại trừ bệnh và không thay thế khám y tế.",
        "info",
    )


def render_supplemental_assessment(values: dict[str, float]) -> None:
    """Show the safety/context result separately from the ML probability."""
    title, body, severity = supplemental_assessment(values)
    if severity == "critical":
        st.error(f"**{title}:** {body}")
    elif severity == "warning":
        st.warning(f"**{title}:** {body}")
    else:
        st.info(f"**{title}:** {body}")


def has_unclear_symptoms(values: dict[str, float]) -> bool:
    return float(values.get("symptom_info_unclear", 0.0)) == 1.0


def blood_pressure_follow_up(systolic: float) -> list[str]:
    """Give actionable, plain-language follow-up for the systolic value only."""
    value = float(systolic)
    measurement = (
        "Đo lại đúng cách: tránh cà phê, thuốc lá và vận động trong 30 phút trước khi đo; "
        "ngồi tựa lưng, đặt hai chân xuống sàn, nghỉ yên ít nhất 5 phút, giữ tay ngang tim "
        "và ghi cả hai số."
    )
    lifestyle = (
        "Để cải thiện sức khỏe và hỗ trợ hạ huyết áp: ăn nhạt hơn (bớt nước mắm, đồ muối, mì gói "
        "và thực phẩm chế biến), ưu tiên rau quả và thực phẩm ít béo, vận động vừa sức đều đặn "
        "khi không có triệu chứng cảnh báo, không hút thuốc, hạn chế rượu, ngủ đủ và dùng thuốc "
        "đúng chỉ định."
    )

    if value < 90:
        return [
            "Huyết áp tâm thu đang thấp theo mốc tham khảo. Ngồi hoặc nằm nghỉ, đứng dậy từ từ và đo lại đủ hai số; không tự dùng thuốc hay thực phẩm bổ sung để nâng huyết áp.",
            "Nếu kèm chóng mặt nhiều, ngất, lú lẫn, đau ngực, khó thở hoặc da lạnh vã mồ hôi, cần được đánh giá y tế ngay.",
            measurement,
        ]
    if value < 120:
        return [
            "Huyết áp tâm thu hiện trong vùng tham khảo. Tiếp tục đo định kỳ và duy trì ăn uống lành mạnh, vận động vừa sức, không hút thuốc và ngủ đủ.",
            measurement,
        ]
    if value < 130:
        return [
            "Huyết áp tâm thu đang cao hơn vùng tham khảo nhưng chưa thuộc mức tăng rõ. Theo dõi nhiều lần vào các ngày khác nhau để biết xu hướng, không kết luận chỉ từ một lần đo.",
            lifestyle,
            measurement,
        ]
    if value < 140:
        return [
            "Huyết áp tâm thu đang tăng nhẹ. Nếu nhiều lần đo vẫn ở mức này, nên đặt lịch trao đổi với nhân viên y tế để đánh giá đầy đủ cả số tâm thu và số tâm trương.",
            lifestyle,
            measurement,
        ]
    if value < 180:
        return [
            "Huyết áp tâm thu đang cao. Nên đặt lịch khám sớm, mang theo nhật ký đo và danh sách thuốc; không tự tăng, giảm hoặc ngừng thuốc.",
            lifestyle,
            measurement,
        ]
    return [
        "Huyết áp tâm thu đang rất cao. Không tập luyện hay tự lái xe; ngồi yên ít nhất 5 phút rồi đo lại đúng kỹ thuật và đo đủ hai số.",
        "Nếu vẫn từ 180 mmHg trở lên (hoặc số dưới từ 120 mmHg trở lên) mà chưa có triệu chứng, liên hệ cơ sở y tế ngay trong ngày để được hướng dẫn.",
        "Nếu có đau ngực, khó thở, đau lưng, tê/yếu một bên, nhìn mờ, khó nói, lú lẫn hoặc ngất, gọi cấp cứu ngay và không chờ ứng dụng.",
        lifestyle,
    ]


def patient_recommendations(
    values: dict[str, float],
    probability: float,
    decision_threshold: float,
) -> list[str]:
    """Create plain-language, safety-oriented follow-up suggestions."""
    recommendations: list[str] = []
    current_warning = has_current_warning(values)
    historical_symptoms = has_chest_symptoms(values)
    unclear_symptoms = has_unclear_symptoms(values)

    if current_warning:
        recommendations.append(
            "<strong>Nếu hiện đang đau ngực, khó thở, vã mồ hôi, choáng/ngất hoặc đau lan ra tay, hàm hay lưng:</strong> "
            "dừng vận động và liên hệ cấp cứu ngay; không chờ kết quả của ứng dụng."
        )
    elif unclear_symptoms:
        recommendations.append(
            "Thông tin về cơn đau ngực chưa đủ rõ để phân loại; nên hỏi lại bệnh nhân hoặc trao đổi với nhân viên y tế thay vì tự chọn một loại đau ngực."
        )
    elif historical_symptoms:
        recommendations.append(
            "Hồ sơ có ghi nhận triệu chứng khi hỏi bệnh hoặc khi gắng sức; nếu triệu chứng đã hết, nên đặt lịch khám sớm, mang theo kết quả và không cố tập nặng để tự kiểm tra."
        )
    else:
        recommendations.append(
            "Hiện chưa ghi nhận kiểu đau ngực rõ trong thông tin nhập; nếu xuất hiện triệu chứng mới hoặc nặng lên, cần đi khám."
        )

    recommendations.extend(blood_pressure_follow_up(float(values["trestbps"])))

    if probability >= decision_threshold:
        recommendations.append(
            "Vì xác suất dự đoán vượt ngưỡng sàng lọc, nên đem kết quả này cùng các xét nghiệm gần nhất để trao đổi với nhân viên y tế."
        )
    else:
        recommendations.append(
            "Nếu hiện không có triệu chứng cảnh báo, có thể theo dõi định kỳ và cập nhật lại hồ sơ khi có số đo hoặc triệu chứng mới."
        )

    recommendations.append(
        "Kết quả ứng dụng chỉ là thông tin hỗ trợ, không thay thế chẩn đoán hoặc kế hoạch điều trị của nhân viên y tế."
    )
    return recommendations


def render_recommendations(
    values: dict[str, float],
    probability: float,
    decision_threshold: float,
) -> None:
    items = "".join(f"<li>{item}</li>" for item in patient_recommendations(values, probability, decision_threshold))
    urgent = ""
    if has_current_warning(values):
        urgent = (
            '<div class="urgent-note"><strong>Lưu ý khẩn:</strong> ứng dụng không thể xác định bệnh nhân có đang trong tình trạng cấp cứu. '
            "Nếu triệu chứng đang xảy ra, hãy gọi cấp cứu địa phương ngay.</div>"
        )
    st.markdown(
        f"""
        <div class="recommendation-card">
            <h3>Nên làm gì tiếp theo?</h3>
            <ul class="recommendation-list">{items}</ul>
            {urgent}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conclusion(
    probability: float,
    level: dict,
    decision_threshold: float,
    values: dict[str, float],
    explanation: pd.DataFrame,
) -> None:
    """Show a plain-language conclusion after all inputs have been evaluated."""
    screening_label = "cần theo dõi thêm" if probability >= decision_threshold else "có thể theo dõi định kỳ"
    top = explanation.iloc[0]
    delta = float(top["delta_probability"])
    top_meaning = feature_value_meaning(top["feature"], top["patient_value"])
    top_detail = f"{top['label_vi']} ({top_meaning})" if top_meaning else str(top["label_vi"])
    if abs(delta) < 0.0005:
        factor_sentence = f"Yếu tố đứng đầu bảng so sánh là {top_detail}, nhưng tác động không đáng kể."
    else:
        top_effect = "tăng" if delta > 0 else "giảm"
        factor_sentence = (
            f"Yếu tố có tác động lớn nhất theo mô hình là {top_detail}, làm {top_effect} "
            f"xác suất khoảng {format_percentage_points_vi(abs(delta))} so với tham chiếu."
        )
    bp_label, _, _ = blood_pressure_assessment(values["trestbps"])
    st.markdown(
        f"""
        <div class="conclusion-card">
            <h3>Kết luận dễ hiểu</h3>
            <p><strong>Bệnh nhân đang như nào?</strong> {clinical_interpretation(probability)} <strong>Kết quả đo:</strong> xác suất dự đoán {format_percent_vi(probability)}, thuộc vùng {level["name"].lower()} và {screening_label} theo ngưỡng {format_percent_vi(decision_threshold)}. {factor_sentence} Huyết áp tâm thu hiện tại là “{bp_label}” theo mốc tham khảo; cần đọc cùng số tâm trương và bối cảnh lâm sàng. Đây là kết luận hỗ trợ sàng lọc, không phải chẩn đoán.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clinical_strip(metadata: dict) -> None:
    distribution = metadata["target_distribution"]
    metrics = metadata.get("test_metrics", {})
    threshold = metadata.get("decision_threshold", metrics.get("threshold", 0.5))
    brier = metrics.get("brier_score")
    brier_text = f"Sai số xác suất {format_decimal_vi(brier, 3)}" if brier is not None else "đã lưu trong báo cáo"
    distribution_text = (
        f'{distribution.get("0", 0)} không có bệnh, {distribution.get("1", 0)} có bệnh '
        "(theo nhãn dữ liệu)"
    )
    with st.expander("Thông tin mô hình và dữ liệu", expanded=False):
        st.markdown(
            f"""
            <div class="clinical-strip">
                <div class="clinical-strip-item"><span>Dữ liệu</span><strong>{metadata.get("dataset_label", "UCI Heart Disease")}, {metadata["n_samples"]} hồ sơ</strong></div>
                <div class="clinical-strip-item"><span>Quy trình</span><strong>Tiền xử lý an toàn + kiểm định chéo 5 phần</strong></div>
                <div class="clinical-strip-item"><span>Phân bố nhãn</span><strong>{distribution_text}</strong></div>
                <div class="clinical-strip-item"><span>Mốc sàng lọc</span><strong>{format_decimal_vi(threshold, 2)} sau hiệu chỉnh; {brier_text}</strong></div>
            </div>
            <p class="medical-note">Các thông tin này mô tả chất lượng và nguồn dữ liệu của mô hình, không phải kết quả riêng của bệnh nhân.</p>
            """,
            unsafe_allow_html=True,
        )


def render_patient_snapshot(values: dict[str, float], patient_name: str = "") -> None:
    diastolic = values.get("trestbps_diastolic")
    blood_pressure_value = display_value("trestbps", values["trestbps"])
    blood_pressure_label = "Huyết áp tâm thu (số trên)"
    chest_pain_value = "Chưa đủ thông tin" if has_unclear_symptoms(values) else display_value("cp", values["cp"])
    if diastolic is not None:
        blood_pressure_value = (
            f"{format_decimal_vi(values['trestbps'], 0)}/{format_decimal_vi(diastolic, 0)} mmHg"
        )
        blood_pressure_label = "Huyết áp"
    cards = [
        *(([("Tên bệnh nhân", escape(patient_name))] if patient_name else [])),
        ("Tuổi", display_value("age", values["age"])),
        ("Giới tính", display_value("sex", values["sex"])),
        ("Đặc điểm đau ngực", chest_pain_value),
        ("Triệu chứng lúc này", current_symptom_label(values.get("current_symptoms", 0.0))),
        (blood_pressure_label, blood_pressure_value),
        ("Cholesterol toàn phần", display_value("chol", values["chol"])),
        ("Nhịp tim cao nhất", display_value("thalach", values["thalach"])),
    ]
    html = ['<div class="patient-grid">']
    for label, value in cards:
        html.append(f'<div class="patient-card"><span>{label}</span><strong>{value}</strong></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_save_record(
    patient_name: str,
    values: dict[str, float],
    probability: float,
    level: dict,
    decision_threshold: float,
) -> None:
    st.markdown('<p class="section-title">Lưu hồ sơ</p>', unsafe_allow_html=True)
    st.caption(f"Dữ liệu được lưu cục bộ tại: {PATIENT_RECORDS_PATH}")
    if st.button(
        "Lưu hồ sơ và kết quả",
        type="primary",
        use_container_width=True,
        key="save_patient_record",
    ):
        if not patient_name:
            st.warning("Vui lòng nhập tên bệnh nhân trước khi lưu hồ sơ.")
            return
        screening_label = "Cần theo dõi thêm" if probability >= decision_threshold else "Theo dõi định kỳ"
        record_id = save_patient_record(
            patient_name=patient_name,
            values=values,
            probability=probability,
            risk_level=level["name"],
            decision_threshold=decision_threshold,
            screening_label=screening_label,
        )
        st.success(f"Đã lưu hồ sơ của {patient_name} (mã: {record_id[:8]}).")


def render_saved_records() -> None:
    st.markdown('<p class="section-title">Hồ sơ đã lưu</p>', unsafe_allow_html=True)
    st.caption(
        "Các bản ghi được lưu trên máy đang chạy ứng dụng. Không nên dùng file này làm hệ thống hồ sơ bệnh án chính thức."
    )
    records = load_patient_records()
    if records.empty:
        st.info("Chưa có hồ sơ nào được lưu. Hãy phân tích một hồ sơ rồi bấm “Lưu hồ sơ và kết quả”.")
        return

    display = records[
        ["saved_at", "patient_name", "probability", "risk_level", "screening_label", "record_id"]
    ].copy()
    display["probability"] = display["probability"].map(lambda value: format_percent_vi(value))
    display = display.rename(
        columns={
            "saved_at": "Thời gian lưu",
            "patient_name": "Tên bệnh nhân",
            "probability": "Xác suất",
            "risk_level": "Mức dự đoán",
            "screening_label": "Theo dõi",
            "record_id": "Mã hồ sơ",
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button(
        "Tải toàn bộ dữ liệu đã lưu (CSV)",
        data=records.to_csv(index=False).encode("utf-8-sig"),
        file_name="patient_records.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_patient_records",
    )


def risk_gauge(probability: float, color: str, decision_threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=probability * 100,
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickfont": {"size": 13},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 40], "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.7,
                    "value": decision_threshold * 100,
                },
            },
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.17,
        xref="paper",
        yref="paper",
        text=format_percent_vi(probability),
        showarrow=False,
        font={"size": 34, "color": "#102A43"},
    )
    fig.add_annotation(
        x=0.5,
        y=-0.06,
        xref="paper",
        yref="paper",
        text=f"Vạch đen: mốc sàng lọc {format_percent_vi(decision_threshold)}",
        showarrow=False,
        font={"size": 14, "color": "#475569"},
    )
    fig.update_layout(
        height=290,
        margin=dict(l=18, r=18, t=16, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial", "size": 14, "color": "#172033"},
    )
    return fig


def select_explanation_rows(explanation: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Keep the chart and table focused on materially different factors."""
    ordered = explanation.sort_values("abs_delta", ascending=False)
    meaningful = ordered[ordered["abs_delta"] >= 0.005]
    selected = meaningful if not meaningful.empty else ordered.head(3)
    return selected.head(limit).copy()


def explanation_chart(explanation: pd.DataFrame):
    top = select_explanation_rows(explanation)
    top["delta_points"] = top["delta_probability"] * 100
    top["Hướng tác động"] = top["delta_probability"].apply(
        lambda value: (
            "Tăng dự đoán"
            if float(value) >= 0.005
            else "Giảm dự đoán"
            if float(value) <= -0.005
            else "Không đáng kể"
        )
    )
    top["Giá trị hiện tại"] = top.apply(
        lambda row: display_value(row["feature"], row["patient_value"]), axis=1
    )
    top["Giá trị tham chiếu"] = top.apply(
        lambda row: display_value(row["feature"], row["reference_value"]), axis=1
    )
    top["Mô tả yếu tố"] = top["feature"].map(
        lambda feature: FEATURE_INFO[feature].get("description_vi", "")
    )
    top["Tác động hiển thị"] = top["delta_probability"].apply(
        format_delta_vi
    )
    top = top.sort_values("delta_points")
    min_points = float(top["delta_points"].min())
    max_points = float(top["delta_points"].max())
    if min_points >= 0:
        axis_range = [-max(1.0, max_points * 0.12), max(1.0, max_points * 1.25)]
    elif max_points <= 0:
        axis_range = [min(-1.0, min_points * 1.25), max(1.0, abs(min_points) * 0.12)]
    else:
        max_abs_points = max(abs(min_points), abs(max_points), 1.0)
        axis_range = [-max_abs_points * 1.15, max_abs_points * 1.15]
    fig = px.bar(
        top,
        x="delta_points",
        y="label_vi",
        orientation="h",
        color="Hướng tác động",
        text="Tác động hiển thị",
        custom_data=["Giá trị hiện tại", "Giá trị tham chiếu", "Mô tả yếu tố", "delta_points"],
        color_discrete_map={
            "Tăng dự đoán": "#DC2626",
            "Giảm dự đoán": "#0F766E",
            "Không đáng kể": "#64748B",
        },
        labels={"delta_points": "Thay đổi dự đoán (điểm phần trăm)", "label_vi": ""},
    )
    fig.update_layout(
        height=max(300, 76 + 48 * len(top)),
        margin=dict(l=205, r=145, t=48, b=50),
        showlegend=top["Hướng tác động"].nunique() > 1,
        legend_title_text="",
        legend=dict(
            orientation="h",
            x=0,
            y=1.16,
            xanchor="left",
            yanchor="bottom",
            font={"size": 12},
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial", "size": 14, "color": "#172033"},
    )
    fig.update_yaxes(automargin=True)
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Giá trị hiện tại: %{customdata[0]}<br>"
            "Giá trị tham chiếu: %{customdata[1]}<br>"
            "Giải thích: %{customdata[2]}<br>"
            "Thay đổi: %{customdata[3]:+.1f} điểm phần trăm<extra></extra>"
        ),
        textfont={"size": 12, "color": "#334155"},
    )
    # Với thanh âm, nhãn đặt bên ngoài sẽ lấn sang vùng nhãn tên yếu tố.
    # Đặt nhãn vào trong thanh xanh để giữ khoảng cách và vẫn đọc được giá trị.
    for trace in fig.data:
        if trace.name == "Giảm dự đoán":
            trace.textposition = "inside"
            trace.insidetextanchor = "middle"
            trace.textfont = {"size": 12, "color": "#FFFFFF"}
    fig.update_xaxes(
        range=axis_range,
        tickformat="+.0f",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#94A3B8",
        title="Thay đổi dự đoán (điểm phần trăm)",
    )
    return fig


def render_header(metadata: dict) -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <div>
                <p class="eyebrow">Hệ thống hỗ trợ phân tích hồ sơ tim mạch</p>
                <h1>Trợ lý Tim Mạch</h1>
                <p>Nhập thông tin triệu chứng và chỉ số đã có để nhận một đánh giá tham khảo, lời giải thích dễ hiểu và hướng theo dõi tiếp theo.</p>
            </div>
            <div class="hero-stats">
                <div class="stat"><span>Chức năng</span><strong>Phân tích hồ sơ</strong></div>
                <div class="stat"><span>Đầu vào</span><strong>{len(FEATURES)} yếu tố</strong></div>
                <div class="stat"><span>Giới hạn</span><strong>Không thay chẩn đoán</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(model, metadata: dict, values: dict[str, float]) -> None:
    probability = predict_probability(model, values)
    level = risk_level(probability)
    explanation = explain_by_reference_delta(model, values, metadata["reference_values"])
    decision_threshold = metadata.get("decision_threshold", metadata.get("test_metrics", {}).get("threshold", 0.5))
    screening_positive = probability >= decision_threshold
    screening_label = "Cần theo dõi thêm" if screening_positive else "Theo dõi định kỳ"
    screening_color = "#DC2626" if screening_positive else "#0F766E"

    st.markdown('<p class="section-title">Kết quả đánh giá</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-summary" style="--risk-color:{level["color"]};">
            <div>
                <span>Đánh giá sơ bộ theo mô hình</span>
                <h2>{level["name"]}</h2>
                <p><strong>Bệnh nhân đang như nào?</strong> {clinical_interpretation(probability)}</p>
                <p>Kết quả dựa trên hồ sơ đã nhập; không phải chẩn đoán và cần đọc cùng triệu chứng thực tế.</p>
            </div>
            <div class="probability-pill">{format_percent_vi(probability)}</div>
        </div>
        <div class="risk-scale">
            <div class="scale-segment" style="background:#16A34A;"></div>
            <div class="scale-segment" style="background:#D97706;"></div>
            <div class="scale-segment" style="background:#DC2626;"></div>
        </div>
        <div class="scale-caption">
            <span>Mức dự đoán thấp: 0–40%</span>
            <span>Mức dự đoán trung bình: 40–70%</span>
            <span>Mức dự đoán cao: 70–100%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        risk_gauge(probability, level["color"], decision_threshold),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown(
        f"""
        <div class="result-metrics">
            <div class="result-metric"><span>Mốc sàng lọc</span><strong>{format_percent_vi(decision_threshold)}</strong></div>
            <div class="result-metric"><span>Việc nên làm</span><strong style="color:{screening_color};">{screening_label}</strong></div>
            <div class="result-metric"><span>Cách đọc</span><strong>Đọc cùng triệu chứng</strong></div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    render_supplemental_assessment(values)
    render_explain_note(
        "Cách đọc nhanh",
         f"{format_percent_vi(probability)} là xác suất dự đoán của mô hình từ hồ sơ đã nhập. Ngưỡng {format_percent_vi(decision_threshold)} là mốc phân loại dùng để ưu tiên phát hiện sớm trường hợp cần theo dõi; đây không phải ngưỡng chẩn đoán y khoa.",
    )

    render_conclusion(probability, level, decision_threshold, values, explanation)

    increasing = explanation.head(4)
    factor_items = "".join(
        f"<li>{row['label_vi']}: {impact_direction(row['delta_probability']).lower()}, "
        f"{format_percentage_points_vi(abs(row['delta_probability']))}</li>"
        for _, row in increasing.iterrows()
    )
    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <span>Yếu tố đáng chú ý</span>
                <ul class="factor-list">{factor_items}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_recommendations(values, probability, decision_threshold)
    st.markdown(
        '<p class="medical-note">Kết quả này chỉ mang tính hỗ trợ tham khảo trong học tập và sàng lọc ban đầu, không thay thế chẩn đoán hoặc tư vấn của bác sĩ.</p>',
        unsafe_allow_html=True,
    )

    return probability, level, explanation


def render_patient_explanation(explanation: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">Các yếu tố làm thay đổi dự đoán của mô hình</p>', unsafe_allow_html=True)
    selected = select_explanation_rows(explanation)
    directions = set(
        selected["delta_probability"].apply(
            lambda value: "up"
            if value >= 0.005
            else "down"
            if value <= -0.005
            else "neutral"
        )
    )
    if directions == {"up"}:
        note = (
            "Thanh đỏ làm tăng dự đoán. Trong trường hợp này chưa có yếu tố nổi bật làm giảm dự đoán. "
            "Biểu đồ chỉ hiển thị các yếu tố có tác động từ 0,5 điểm phần trăm trở lên. "
            "Đây là giải thích của mô hình, không phải nguyên nhân gây bệnh."
        )
    elif directions == {"down"}:
        note = (
            "Thanh xanh làm giảm dự đoán. Trong trường hợp này chưa có yếu tố nổi bật làm tăng dự đoán. "
            "Biểu đồ chỉ hiển thị các yếu tố có tác động từ 0,5 điểm phần trăm trở lên. "
            "Đây là giải thích của mô hình, không phải nguyên nhân gây bệnh."
        )
    else:
        note = (
            "Thanh đỏ làm tăng dự đoán, thanh xanh làm giảm dự đoán. Giá trị càng xa 0 thì tác động lên "
            "dự đoán càng lớn. Biểu đồ chỉ hiển thị các yếu tố có tác động từ 0,5 điểm phần trăm trở lên. "
            "Đây là giải thích của mô hình, không phải nguyên nhân gây bệnh."
        )
    render_explain_note(
        "Cách đọc biểu đồ",
        note,
    )
    st.plotly_chart(explanation_chart(explanation), width="stretch", config={"displayModeBar": False})

    readable = selected.copy()
    readable["Giá trị hiện tại"] = readable.apply(
        lambda row: display_value(row["feature"], row["patient_value"]), axis=1
    )
    readable["Giá trị tham chiếu"] = readable.apply(
        lambda row: display_value(row["feature"], row["reference_value"]), axis=1
    )
    readable["Tác động"] = readable["delta_probability"].apply(format_delta_vi)
    readable = readable.rename(columns={"label_vi": "Yếu tố", "direction": "Hướng tác động"})
    readable["Hướng tác động"] = readable["delta_probability"].apply(impact_direction)
    readable["Diễn giải"] = readable.apply(
        lambda row: impact_explanation(
            row["delta_probability"],
            row["feature"],
            row["patient_value"],
            row["reference_value"],
        ),
        axis=1,
    )
    render_table(
        readable[["Yếu tố", "Giá trị hiện tại", "Giá trị tham chiếu", "Tác động", "Hướng tác động", "Diễn giải"]]
    )


def interval_text(intervals: dict, metric: str, *, percent: bool = True) -> str:
    interval = intervals.get(metric)
    if not interval:
        return "Chưa có khoảng ước tính"
    if not percent:
        return (
            "Khoảng 95%: "
            f"{format_decimal_vi(interval['lower'], 3)}–"
            f"{format_decimal_vi(interval['upper'], 3)}"
        )
    return (
        f"Khoảng 95%: {format_percent_vi(interval['lower'])}–"
        f"{format_percent_vi(interval['upper'])}"
    )


def render_validation_summary(metadata: dict) -> None:
    confusion = metadata.get("confusion_matrix", [[0, 0], [0, 0]])
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    test_size = metadata.get("split", {}).get("test", sum(sum(row) for row in confusion))
    metrics = metadata.get("test_metrics", {})
    intervals = metadata.get("test_metric_intervals", {})
    st.markdown('<p class="section-title">Độ ổn định trên tập kiểm tra</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="validation-summary">
            <div class="validation-card"><span>Tập kiểm tra</span><strong>{test_size} hồ sơ</strong><small>Đánh giá nội bộ, chưa phải dữ liệu bên ngoài.</small></div>
            <div class="validation-card"><span>Đúng nhóm có bệnh</span><strong>{true_positive}</strong><small>Ca dương tính được phát hiện đúng.</small></div>
            <div class="validation-card"><span>Báo nhầm</span><strong>{false_positive}</strong><small>Hồ sơ không bệnh nhưng bị báo thuộc nhóm có bệnh.</small></div>
            <div class="validation-card"><span>Bỏ sót</span><strong>{false_negative}</strong><small>Hồ sơ có bệnh nhưng mô hình không phát hiện.</small></div>
        </div>
        <div class="validation-note">
            <strong>Đọc thận trọng:</strong> độ nhạy hiện tại {format_percent_vi(metrics.get("recall", 0))},
            độ đặc hiệu {format_percent_vi(metrics.get("specificity", 0))}, AUC {format_decimal_vi(metrics.get("roc_auc", 0), 3)}.
            {interval_text(intervals, "roc_auc")}; {interval_text(intervals, "precision")};
            Brier {format_decimal_vi(metrics.get("brier_score", 0), 3)} ({interval_text(intervals, "brier_score", percent=False)}).
            Khoảng này chỉ phản ánh độ dao động khi lấy mẫu lại tập kiểm tra, không thay thế kiểm định trên dữ liệu độc lập.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_analysis(metadata: dict) -> None:
    st.markdown('<p class="section-title">Cách hiểu chỉ số đánh giá</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="guide-grid">
            <div class="guide-card"><strong>Độ nhạy</strong><span>Trong các hồ sơ thật sự có nhãn bệnh, mô hình phát hiện đúng được bao nhiêu. Chỉ số này giúp theo dõi nguy cơ bỏ sót trong dữ liệu đánh giá.</span></div>
            <div class="guide-card"><strong>Độ chính xác cảnh báo</strong><span>Trong các hồ sơ hệ thống báo thuộc nhóm có bệnh, có bao nhiêu trường hợp đúng theo nhãn dữ liệu. Chỉ số này giúp hiểu mức cảnh báo nhầm.</span></div>
            <div class="guide-card"><strong>AUC ROC</strong><span>Cho biết mô hình xếp hạng hai nhóm có và không có nhãn bệnh tốt đến đâu. Giá trị càng gần 1 thì khả năng phân biệt trên dữ liệu đánh giá càng cao.</span></div>
            <div class="guide-card"><strong>Điểm F2</strong><span>Cân bằng giữa phát hiện đúng và cảnh báo đúng, nhưng đặt nặng phát hiện đúng hơn. Phù hợp bài toán sàng lọc bệnh.</span></div>
            <div class="guide-card"><strong>Hiệu chỉnh xác suất</strong><span>Làm cho xác suất dự đoán đáng tin hơn, không chỉ đơn thuần là điểm số của mô hình.</span></div>
            <div class="guide-card"><strong>Mốc cần theo dõi</strong><span>Đây là mốc để quyết định khi nào hệ thống khuyên theo dõi thêm. Mốc này không phải ngưỡng chẩn đoán và không dùng thay cho bác sĩ.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_validation_summary(metadata)

    st.markdown('<p class="section-title">Bảng so sánh mô hình</p>', unsafe_allow_html=True)
    metrics = load_metrics()
    preferred_columns = [
        "model",
        "selected_by_cv",
        "cv_recall",
        "cv_f1_score",
        "cv_roc_auc",
        "cv_selection_score",
        "recall",
        "roc_auc",
        "average_precision",
    ]
    display = metrics[[column for column in preferred_columns if column in metrics.columns]].copy()
    if "model" in display.columns:
        display["model"] = display["model"].map(model_name_vi)
    if "selected_by_cv" in display.columns:
        display["selected_by_cv"] = display["selected_by_cv"].map(yes_no_vi)
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    display = display.rename(columns=METRIC_LABELS_VI)
    render_table(display)

    threshold_table = load_threshold_analysis()
    if not threshold_table.empty:
        st.markdown('<p class="section-title">Các ngưỡng phân loại tốt nhất theo điểm F2</p>', unsafe_allow_html=True)
        render_explain_note(
            "Vì sao có bảng mốc sàng lọc?",
            "Một mô hình có thể cho ra xác suất, nhưng cần chọn một mốc để quyết định khi nào báo cần theo dõi thêm. Mốc này là lựa chọn kỹ thuật của bài, không phải ngưỡng chẩn đoán y khoa.",
        )
        threshold_display = (
            threshold_table.sort_values(["f2_score", "recall", "precision"], ascending=False)
            .head(8)[["threshold", "precision", "recall", "specificity", "f2_score"]]
            .copy()
        )
        for column in threshold_display.columns:
            threshold_display[column] = threshold_display[column].map(lambda value: f"{value:.3f}")
        threshold_display = threshold_display.rename(columns=THRESHOLD_LABELS_VI)
        render_table(threshold_display)

    st.markdown('<p class="section-title">Biểu đồ đánh giá</p>', unsafe_allow_html=True)
    images = [
        ("So sánh mô hình", FIGURES_DIR / "model_comparison.png"),
        ("Ma trận nhầm lẫn", FIGURES_DIR / "confusion_matrix.png"),
        ("Đường cong khả năng phân biệt", FIGURES_DIR / "roc_curve.png"),
        ("Đường cong độ chính xác - độ nhạy", FIGURES_DIR / "precision_recall_curve.png"),
        ("Tối ưu mốc sàng lọc", FIGURES_DIR / "threshold_tuning.png"),
        ("Hiệu chỉnh xác suất", FIGURES_DIR / "calibration_curve.png"),
        ("Mức ảnh hưởng yếu tố", FIGURES_DIR / "feature_importance.png"),
        ("Xu hướng ảnh hưởng của yếu tố", FIGURES_DIR / "partial_dependence.png"),
    ]
    shap_path = metadata.get("shap_summary_path")
    if metadata.get("shap_model_name") and shap_path and Path(shap_path).exists():
        images.append((f"Tóm tắt SHAP của {model_name_vi(metadata['shap_model_name'])}", Path(shap_path)))
    first, second = st.columns(2)
    for index, (caption, path) in enumerate(images):
        target = first if index % 2 == 0 else second
        with target:
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")


def render_data_overview(metadata: dict) -> None:
    st.markdown('<p class="section-title">Thông tin bộ dữ liệu</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu", metadata["n_samples"])
    c2.metric("Số đặc trưng", metadata["n_features"])
    c3.metric("Không có bệnh (nhãn dữ liệu)", metadata["target_distribution"].get("0", 0))
    c4.metric("Có bệnh (nhãn dữ liệu)", metadata["target_distribution"].get("1", 0))

    st.markdown(
        f"""
        Bộ dữ liệu được lấy từ [UCI Heart Disease Dataset]({metadata["dataset_page"]}).
        Nhãn gốc gồm 5 mức `0-4`, trong đó `0` là không có bệnh và `1-4` là có bệnh theo nhãn của bộ dữ liệu.
        Project chuyển thành bài toán nhị phân để phân loại sự hiện diện bệnh trong dữ liệu, không phải dự đoán nguy cơ mắc bệnh trong tương lai.
        """
    )
    render_explain_note(
        "Các khả năng đầu ra đã đủ chưa?",
        "Mô hình trả về xác suất dự đoán từ 0% đến 100%, sau đó giao diện chia thành 3 vùng hiển thị: thấp, trung bình và cao. Ngoài ra hệ thống có một mốc sàng lọc riêng; nếu vượt mốc thì báo cần theo dõi thêm. Ba vùng hiển thị và mốc này phục vụ tham khảo trong phạm vi mô hình, không phải chẩn đoán.",
    )
    render_explain_note(
        "Các yếu tố đầu vào đã đúng chưa?",
        f"Mô hình hiện dùng {len(FEATURES)} yếu tố đã được giữ lại sau khi đối chiếu bốn nguồn UCI: tuổi, giới tính, kiểu đau ngực, huyết áp, cholesterol, điện tâm đồ lúc nghỉ, nhịp tim tối đa, đau khi vận động và thay đổi đoạn ST. Các biến thiếu quá nhiều như đường huyết lúc đói, hướng ST, số mạch chính và kết quả thallium đã bị loại khỏi đầu vào. Đây chưa phải toàn bộ yếu tố y khoa ngoài đời như hút thuốc, BMI, tiền sử gia đình hoặc thuốc đang dùng; hệ thống chỉ phân tích hồ sơ tham khảo.",
    )

    if FEATURE_FILTER_REPORT_PATH.exists():
        st.markdown('<p class="section-title">Kiểm tra biến trước khi huấn luyện</p>', unsafe_allow_html=True)
        filter_report = pd.read_csv(FEATURE_FILTER_REPORT_PATH)
        filter_report["included_in_model"] = filter_report["included_in_model"].map(
            {True: "Giữ", False: "Loại"}
        ).fillna(filter_report["included_in_model"].astype(str))
        filter_report = filter_report.rename(
            columns={
                "feature": "Mã biến",
                "included_in_model": "Trạng thái",
                "overall_missing_rate": "Tỷ lệ thiếu toàn bộ",
                "max_source_missing_rate": "Tỷ lệ thiếu cao nhất ở một nguồn",
                "reason": "Lý do",
            }
        )
        for column in ["Tỷ lệ thiếu toàn bộ", "Tỷ lệ thiếu cao nhất ở một nguồn"]:
            filter_report[column] = filter_report[column].map(lambda value: f"{float(value):.1%}")
        render_table(
            filter_report[
                [
                    "Mã biến",
                    "Trạng thái",
                    "Tỷ lệ thiếu toàn bộ",
                    "Tỷ lệ thiếu cao nhất ở một nguồn",
                    "Lý do",
                ]
            ]
        )

    if DATASET_SOURCES_PATH.exists():
        source_report = pd.read_csv(DATASET_SOURCES_PATH)
        render_explain_note(
            "Phân bố theo nguồn UCI",
            "Các nguồn có cách thu thập khác nhau; vì vậy kết quả đánh giá nội bộ không thay thế kiểm định trên bệnh viện hoặc nhóm bệnh nhân mới.",
        )
        render_table(source_report)

    st.markdown('<p class="section-title">Từ điển đặc trưng</p>', unsafe_allow_html=True)
    rows = []
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        rows.append(
            {
                "Mã biến": feature,
                "Tên hiển thị": info["ui_label"],
                "Loại": "Liên tục" if info["kind"] == "continuous" else "Phân loại",
                "Đơn vị": info.get("unit", ""),
                "Ý nghĩa ngắn": FEATURE_HELP.get(feature, info.get("description_vi", "")),
            }
        )
    render_table(pd.DataFrame(rows))

    st.markdown('<p class="section-title">Khám phá dữ liệu</p>', unsafe_allow_html=True)
    first, second = st.columns(2)
    with first:
        st.image(str(FIGURES_DIR / "target_distribution.png"), caption="Phân bố nhãn", width="stretch")
        st.image(str(FIGURES_DIR / "feature_distributions.png"), caption="Phân bố một số đặc trưng", width="stretch")
    with second:
        st.image(str(FIGURES_DIR / "correlation_heatmap.png"), caption="Ma trận tương quan", width="stretch")
        group_mean_path = FIGURES_DIR / "group_mean_comparison.png"
        if group_mean_path.exists():
            st.image(str(group_mean_path), caption="So sánh trung bình theo nhóm", width="stretch")

    st.markdown('<p class="section-title">Yếu tố ảnh hưởng toàn cục</p>', unsafe_allow_html=True)
    importance = load_feature_importance().head(10).copy()
    importance = importance[["label_vi", "importance_mean", "importance_std"]].rename(
        columns={
            "label_vi": "Yếu tố",
            "importance_mean": "Mức ảnh hưởng trung bình",
            "importance_std": "Độ dao động",
        }
    )
    importance["Mức ảnh hưởng trung bình"] = importance["Mức ảnh hưởng trung bình"].map(lambda value: f"{value:.4f}")
    importance["Độ dao động"] = importance["Độ dao động"].map(lambda value: f"{value:.4f}")
    render_explain_note(
        "Cách hiểu bảng yếu tố ảnh hưởng",
        "Mức ảnh hưởng càng lớn thì khi xáo trộn yếu tố đó, chất lượng dự đoán càng giảm nhiều. Vì vậy yếu tố đó được xem là quan trọng hơn đối với mô hình.",
    )
    render_table(importance)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Chưa có model. Hãy chạy lệnh: python -m src.train_model")
        return

    model, metadata = cached_artifacts()
    render_header(metadata)
    render_clinical_strip(metadata)

    tab_predict, tab_history, tab_model, tab_data = st.tabs(
        ["Phân tích hồ sơ", "Hồ sơ đã lưu", "Hiệu năng mô hình", "Dữ liệu và giải thích"]
    )

    with tab_predict:
        form_col, result_col = st.columns([1.08, 0.92], gap="large")
        with form_col:
            with st.container(border=False):
                patient_name, values = collect_patient_inputs()
                st.markdown('<p class="section-title">Tóm tắt hồ sơ</p>', unsafe_allow_html=True)
                render_patient_snapshot(values, patient_name)
                st.markdown('<p class="section-title">Kiểm tra nhanh hồ sơ</p>', unsafe_allow_html=True)
                render_explain_note(
                    "Cách đọc các tín hiệu",
                    "Màu đỏ là tình huống cần chú ý sớm, màu cam là nên theo dõi hoặc trao đổi thêm, màu xanh là thông tin tham khảo. Đây không phải kết luận y khoa độc lập.",
                )
                render_patient_flags(values)
        with result_col:
            probability, level, explanation = render_result(model, metadata, values)
            decision_threshold = metadata.get(
                "decision_threshold",
                metadata.get("test_metrics", {}).get("threshold", 0.5),
            )
            render_save_record(patient_name, values, probability, level, decision_threshold)

        with st.expander("Xem giải thích chi tiết của mô hình", expanded=False):
            render_patient_explanation(explanation)

    with tab_history:
        render_saved_records()

    with tab_model:
        render_model_analysis(metadata)

    with tab_data:
        render_data_overview(metadata)


if __name__ == "__main__":
    main()
