from __future__ import annotations

import os
from pathlib import Path

# Base paths used by both pipeline and app for loading data and writing outputs.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_set"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs")))

# Canonical output filenames shared across scripts.
MODEL_RESULTS_FILE = "model_results.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
OUTPUT_FILES = {
    "model_results": MODEL_RESULTS_FILE,
    "feature_importance": FEATURE_IMPORTANCE_FILE,
}

# Core training/evaluation parameters that affect reproducibility and metrics.
MODEL_CONFIG = {
    "random_state": 42,
    "train_split_test_size": 0.30,
    "val_test_split_size": 0.50,
    "cv_folds": 5,
    "perm_repeats": 30,
}

# Class names used in predictions, reports, and evaluation outputs.
CLASS_NAMES = ["Low Stress", "Medium Stress", "High Stress"]
APP_CLASS_COLOURS = ["#27ae60", "#2980b9", "#c0392b"]
APP_CLASS_BG = ["#d5f5e3", "#d6eaf8", "#fadbd8"]
PIPELINE_CLASS_COLOURS = ["#55A868", "#4C72B0", "#C44E52"]

# Mapping from survey text buckets to numeric hours for model-ready features.
SM_HOURS_MAP = {
    "Less than an Hour": 0.5,
    "Between 1 and 2 hours": 1.5,
    "Between 2 and 3 hours": 2.5,
    "Between 3 and 4 hours": 3.5,
    "Between 4 and 5 hours": 4.5,
    "More than 5 hours": 5.5,
}

# Dataset provenance links used for traceability in UI/report sections.
DATASET_LINKS = {
    "DS1 — Mobile Device Usage and User Behavior": "https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset?resource=download",
    "DS2 — Social Media and Mental Health": "https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health",
    "DS3 — Remote Work and Mental Health": "https://www.kaggle.com/datasets/waqi786/remote-work-and-mental-health",
}

# App navigation/state constants used by page routing and loading behavior.
PAGE_OVERVIEW = "Overview"
PAGE_STRESS_PREDICTOR = "Stress Predictor"
PAGE_DEVICE_PREDICTIONS = "Device Predictions"
PAGE_PROJECT_REPORT = "Project Report"
NAV_PAGES = [PAGE_OVERVIEW, PAGE_STRESS_PREDICTOR, PAGE_DEVICE_PREDICTIONS, PAGE_PROJECT_REPORT]
MIN_PAGE_LOADER_SECONDS = 0.35

# UI styling tokens (kept centralized for quick theme updates).
UI_THEME = {
    "sidebar_bg": "#0d1117",
    "sidebar_primary": "#1f6feb",
    "sidebar_border": "#30363d",
    "muted_text": "#8b949e",
    "panel_bg": "#161b22",
}

LAYOUT_CONFIG = {
    "sidebar_btn_margin": "2px 0",
    "result_margin": "1rem 0",
    "section_heading_margin": "1.2rem 0 0.6rem",
    "plot_margin": {"l": 12, "r": 12, "t": 36, "b": 12},
}

# Raw DS2 survey column names mapped to concise ML-friendly names.
DS2_COL_MAP = {
    "1. What is your age?": "Age",
    "2. Gender": "Gender",
    "3. Relationship Status": "Relationship_Status",
    "4. Occupation Status": "Occupation",
    "8. What is the average time you spend on social media every day?": "Social_Media_Hours",
    "9. How often do you find yourself using Social media without a specific purpose?": "Purposeless_Use",
    "10. How often do you get distracted by Social media when you are busy doing something?": "Distraction_Score",
    "11. Do you feel restless if you haven't used Social media in a while?": "Restlessness",
    "12. On a scale of 1 to 5, how easily distracted are you?": "Easily_Distracted",
    "13. On a scale of 1 to 5, how much are you bothered by worries?": "Worry_Score",
    "14. Do you find it difficult to concentrate on things?": "Difficulty_Concentrating",
    "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?": "Comparison_Score",
    "17. How often do you look to seek validation from features of social media?": "Validation_Seeking",
    "18. How often do you feel depressed or down?": "Depression_Score",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": "Interest_Fluctuation",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?": "Sleep_Issues",
}

# Inputs used to compute the engineered Composite_Stress target.
STRESS_INDICATOR_COLS = [
    "Worry_Score",
    "Depression_Score",
    "Sleep_Issues",
    "Restlessness",
    "Easily_Distracted",
    "Difficulty_Concentrating",
    "Validation_Seeking",
    "Interest_Fluctuation",
]

# Final ordered feature vector used for scaling/training/prediction.
FEATURE_COLS = [
    "Social_Media_Hours",
    "Purposeless_Use",
    "Distraction_Score",
    "Restlessness",
    "Easily_Distracted",
    "Worry_Score",
    "Difficulty_Concentrating",
    "Comparison_Score",
    "Validation_Seeking",
    "Depression_Score",
    "Interest_Fluctuation",
    "Sleep_Issues",
    "Age",
    "Gender_enc",
    "Relationship_Status_enc",
    "Occupation_enc",
    "Avg_Screen_Time_hrs",
    "Avg_Battery_Drain_mAh",
    "Avg_App_Usage_min",
    "Avg_Data_Usage_MB",
    "Avg_Apps_Installed",
    "Avg_Mobile_Intensity",
    "Avg_Work_Hours_Week",
    "Avg_Work_Life_Balance",
    "Avg_Social_Isolation",
    "Avg_Sleep_Quality",
    "Avg_Virtual_Meetings",
]
