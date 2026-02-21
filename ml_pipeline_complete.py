"""ml_pipeline_complete.py

Real-world Problem: Digital Wellbeing Stress Prediction
Algorithm: Gaussian Naive Bayes (GNB)

Notes:
- This script is written as a single end-to-end pipeline.
- It reads 3 CSV datasets from the local `data_set/` folder.
- It creates an engineered 3-class stress label (Low/Medium/High).
- It trains and evaluates a Gaussian Naive Bayes model.
- It saves plots and CSV outputs to an `outputs/` folder (configurable).

Student: 214129X — Malalpola MLHR
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_set"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Low Stress", "Medium Stress", "High Stress"]


def print_step(title: str) -> None:
    bar = "=" * 60
    print("\n" + bar)
    print(title)
    print(bar)


def assign_age_group(age: float) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def load_datasets(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df1_path = data_dir / "user_behavior_dataset.csv"
    df2_path = data_dir / "smmh.csv"
    df3_path = data_dir / "Impact_of_Remote_Work_on_Mental_Health.csv"

    if not df1_path.exists() or not df2_path.exists() or not df3_path.exists():
        missing = [
            str(p) for p in [df1_path, df2_path, df3_path] if not p.exists()
        ]
        raise FileNotFoundError(
            "One or more input CSV files were not found:\n" + "\n".join(missing)
        )

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df3 = pd.read_csv(df3_path)
    return df1, df2, df3


def preprocess_primary_dataset(df2_raw: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Preprocess DS2 (primary dataset) and engineer the stress label."""
    col_map = {
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

    df2 = df2_raw.rename(columns=col_map).copy()

    sm_map = {
        "Less than an Hour": 0.5,
        "Between 1 and 2 hours": 1.5,
        "Between 2 and 3 hours": 2.5,
        "Between 3 and 4 hours": 3.5,
        "Between 4 and 5 hours": 4.5,
        "More than 5 hours": 5.5,
    }
    df2["Social_Media_Hours"] = df2["Social_Media_Hours"].map(sm_map)
    df2["Age"] = pd.to_numeric(df2["Age"], errors="coerce")
    df2 = df2.dropna(subset=["Age", "Social_Media_Hours"]).copy()

    def encode_series(series: pd.Series) -> np.ndarray:
        encoder = LabelEncoder()
        cleaned = series.astype(str).str.strip()
        return encoder.fit_transform(cleaned)

    df2["Gender_enc"] = encode_series(df2["Gender"])
    df2["Relationship_Status_enc"] = encode_series(df2["Relationship_Status"])
    df2["Occupation_enc"] = encode_series(df2["Occupation"])

    # Engineer composite stress target
    stress_indicators = [
        "Worry_Score",
        "Depression_Score",
        "Sleep_Issues",
        "Restlessness",
        "Easily_Distracted",
        "Difficulty_Concentrating",
        "Validation_Seeking",
        "Interest_Fluctuation",
    ]
    df2["Composite_Stress"] = df2[stress_indicators].mean(axis=1)

    t33 = float(df2["Composite_Stress"].quantile(0.33))
    t66 = float(df2["Composite_Stress"].quantile(0.66))

    def classify_stress(score: float) -> int:
        if score <= t33:
            return 0
        if score <= t66:
            return 1
        return 2

    df2["Stress_Label"] = df2["Composite_Stress"].apply(classify_stress)
    return df2, t33, t66


def build_ds1_enrichment(df1: pd.DataFrame) -> pd.DataFrame:
    df1_agg = df1.copy()
    df1_agg["Age_Group"] = df1_agg["Age"].apply(assign_age_group)
    ds1_enriched = (
        df1_agg.groupby("Age_Group")
        .agg(
            Avg_Screen_Time_hrs=("Screen On Time (hours/day)", "mean"),
            Avg_Battery_Drain_mAh=("Battery Drain (mAh/day)", "mean"),
            Avg_App_Usage_min=("App Usage Time (min/day)", "mean"),
            Avg_Data_Usage_MB=("Data Usage (MB/day)", "mean"),
            Avg_Apps_Installed=("Number of Apps Installed", "mean"),
            Avg_Mobile_Intensity=("User Behavior Class", "mean"),
        )
        .reset_index()
    )
    return ds1_enriched


def build_ds3_enrichment(df3: pd.DataFrame) -> pd.DataFrame:
    df3_clean = df3.copy()
    df3_clean["Physical_Activity"] = df3_clean["Physical_Activity"].fillna(
        df3_clean["Physical_Activity"].mode()[0]
    )
    df3_clean = df3_clean.dropna(subset=["Mental_Health_Condition"]).copy()
    df3_clean["Age_Group"] = df3_clean["Age"].apply(assign_age_group)

    sleep_map = {"Poor": 1, "Average": 2, "Good": 3}
    df3_clean["Sleep_Quality_num"] = df3_clean["Sleep_Quality"].map(sleep_map)

    ds3_enriched = (
        df3_clean.groupby("Age_Group")
        .agg(
            Avg_Work_Hours_Week=("Hours_Worked_Per_Week", "mean"),
            Avg_Work_Life_Balance=("Work_Life_Balance_Rating", "mean"),
            Avg_Social_Isolation=("Social_Isolation_Rating", "mean"),
            Avg_Sleep_Quality=("Sleep_Quality_num", "mean"),
            Avg_Virtual_Meetings=("Number_of_Virtual_Meetings", "mean"),
        )
        .reset_index()
    )
    return ds3_enriched


def merge_to_unified_dataset(
    df2: pd.DataFrame, ds1_enriched: pd.DataFrame, ds3_enriched: pd.DataFrame
) -> pd.DataFrame:
    df2 = df2.copy()
    df2["Age_Group"] = df2["Age"].apply(assign_age_group)
    df_main = df2.merge(ds1_enriched, on="Age_Group", how="left", validate="many_to_one")
    df_main = df_main.merge(ds3_enriched, on="Age_Group", how="left", validate="many_to_one")
    df_main = df_main.dropna().copy()
    return df_main


def prepare_features_and_target(
    df_main: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        # DS2 — Social media behaviour
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
        # DS2 — Demographics
        "Age",
        "Gender_enc",
        "Relationship_Status_enc",
        "Occupation_enc",
        # DS1 — Mobile usage enrichment
        "Avg_Screen_Time_hrs",
        "Avg_Battery_Drain_mAh",
        "Avg_App_Usage_min",
        "Avg_Data_Usage_MB",
        "Avg_Apps_Installed",
        "Avg_Mobile_Intensity",
        # DS3 — Work/PC context enrichment
        "Avg_Work_Hours_Week",
        "Avg_Work_Life_Balance",
        "Avg_Social_Isolation",
        "Avg_Sleep_Quality",
        "Avg_Virtual_Meetings",
    ]

    x = df_main[feature_cols].copy()
    y = df_main["Stress_Label"].copy()

    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=feature_cols)
    return x_scaled, y, feature_cols


def split_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=random_state, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_gnb(
    x_train: pd.DataFrame, y_train: pd.Series, *, random_state: int = 42
) -> Tuple[GaussianNB, np.ndarray]:
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(gnb, x_train, y_train, cv=cv, scoring="accuracy")
    return gnb, cv_scores


def evaluate_split(
    model: GaussianNB,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> Dict[str, float | np.ndarray]:
    y_pred = model.predict(x_eval)
    y_proba = model.predict_proba(x_eval)
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average="weighted")
    auc = roc_auc_score(y_eval, y_proba, multi_class="ovr", average="weighted")
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "auc_ovr_weighted": float(auc),
    }


def print_class_distribution(y: pd.Series) -> None:
    dist = dict(y.value_counts().sort_index())
    print(f"Class distribution: {dist}")


def compute_permutation_importance(
    model: GaussianNB,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="accuracy",
    )
    perm_df = (
        pd.DataFrame(
            {
                "Feature": feature_cols,
                "Importance": perm.importances_mean,
                "Std": perm.importances_std,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return perm_df

def generate_all_plots(
    *,
    output_dir: Path,
    df_main: pd.DataFrame,
    y_all: pd.Series,
    feature_cols: List[str],
    cv_scores: np.ndarray,
    val_metrics: Dict[str, float | np.ndarray],
    test_metrics: Dict[str, float | np.ndarray],
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    y_test_proba: np.ndarray,
    perm_df: pd.DataFrame,
    model: GaussianNB,
    x_test: pd.DataFrame,
) -> None:
    print_step("STEP 9 — GENERATING ALL PLOTS")

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_all.value_counts().sort_index()
    bars = ax.bar(
        CLASS_NAMES,
        counts.values,
        color=["#55A868", "#4C72B0", "#C44E52"],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(
        "Target Variable Distribution\nStress Level (Healthy / Moderate / At Risk)",
        fontsize=12,
        fontweight="bold",
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(value),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "plot1_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot1_class_distribution.png")

    # Plot 2: Correlation Heatmap (DS2 features)
    fig, ax = plt.subplots(figsize=(11, 9))
    ds2_feat_corr = [
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
        "Stress_Label",
    ]
    corr_matrix = df_main[ds2_feat_corr].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Feature Correlation Matrix\n(Social Media & Mental Health Features)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "plot2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot2_correlation_heatmap.png")

    # Plot 3: CV Accuracy per Fold
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [f"Fold {i + 1}" for i in range(5)]
    colors = ["#4C72B0" if s >= cv_scores.mean() else "#DD8452" for s in cv_scores]
    bars = ax.bar(folds, cv_scores, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(
        cv_scores.mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {cv_scores.mean():.3f}",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        "5-Fold Stratified Cross-Validation Accuracy\nGaussian Naive Bayes",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    for bar, val in zip(bars, cv_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "plot3_cv_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot3_cv_accuracy.png")

    # Plot 4: Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        "Confusion Matrix — Test Set\nGaussian Naive Bayes",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "plot4_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot4_confusion_matrix.png")

    # Plot 5: ROC Curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_colors = ["#55A868", "#4C72B0", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, roc_colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])
        ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{cls} (AUC = {auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curves — One vs Rest\nGaussian Naive Bayes (Test Set)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "plot5_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot5_roc_curves.png")

    # Plot 6: Performance Metrics Summary
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = [
        "CV Accuracy\n(Train)",
        "Accuracy\n(Val)",
        "F1-Score\n(Val)",
        "Accuracy\n(Test)",
        "F1-Score\n(Test)",
        "AUC\n(Test)",
    ]
    values = [
        float(cv_scores.mean()),
        float(val_metrics["accuracy"]),
        float(val_metrics["f1_weighted"]),
        float(test_metrics["accuracy"]),
        float(test_metrics["f1_weighted"]),
        float(test_metrics["auc_ovr_weighted"]),
    ]
    bar_colors = ["#4C72B0", "#55A868", "#55A868", "#DD8452", "#DD8452", "#C44E52"]
    bars = ax.bar(metrics, values, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Summary\nGaussian Naive Bayes", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "plot6_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot6_metrics_summary.png")

    # Plot 7: XAI — Permutation Feature Importance
    fig, ax = plt.subplots(figsize=(9, 7))
    top15 = perm_df.head(15).copy()
    imp_colors = ["#C44E52" if v > 0 else "#AAAAAA" for v in top15["Importance"]]
    ax.barh(
        top15["Feature"][::-1],
        top15["Importance"][::-1],
        xerr=top15["Std"][::-1],
        color=imp_colors[::-1],
        edgecolor="white",
        linewidth=1,
        capsize=3,
    )
    ax.set_xlabel("Mean Decrease in Accuracy (when feature permuted)", fontsize=10)
    ax.set_title(
        "XAI — Permutation Feature Importance\nGaussian Naive Bayes on Test Set",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / "plot7_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot7_permutation_importance.png")

    # Plot 8: XAI — Partial Dependence Plots
    numerical_direct = [
        "Social_Media_Hours",
        "Worry_Score",
        "Depression_Score",
        "Sleep_Issues",
        "Distraction_Score",
        "Easily_Distracted",
        "Avg_Screen_Time_hrs",
        "Avg_Work_Hours_Week",
        "Age",
    ]
    top_pdp = [f for f in perm_df.head(10)["Feature"].tolist() if f in numerical_direct][:4]
    top_pdp_idx = [feature_cols.index(f) for f in top_pdp]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model,
        x_test,
        features=top_pdp_idx,
        feature_names=feature_cols,
        target=2,
        ax=axes.ravel(),
        random_state=42,
        line_kw={"color": "#C44E52", "linewidth": 2.2},
    )
    fig.suptitle(
        "Partial Dependence Plots — Effect on High Stress Prediction\n(Top Numerical Features)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "plot8_partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot8_partial_dependence.png")

    # Plot 9: GNB Class Means Heatmap (What the model learned)
    top10_feats = perm_df.head(10)["Feature"].tolist()
    top10_idx = [feature_cols.index(f) for f in top10_feats]
    theta_df = pd.DataFrame(
        model.theta_[:, top10_idx],
        index=CLASS_NAMES,
        columns=[f.replace("_", "\n") for f in top10_feats],
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        theta_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Feature Mean"},
    )
    ax.set_title(
        "GNB Learned Class Means — Top 10 Features\n(What the model learned per stress class)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "plot9_gnb_class_means.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot9_gnb_class_means.png")

    # Plot 10: Social Media Hours Distribution by Stress Class
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, col in zip([0, 1, 2], CLASS_NAMES, ["#55A868", "#4C72B0", "#C44E52"]):
        subset = df_main[df_main["Stress_Label"] == label]["Social_Media_Hours"]
        ax.hist(subset, bins=8, alpha=0.6, label=name, color=col, edgecolor="white")
    ax.set_xlabel("Daily Social Media Hours", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Social Media Usage Distribution by Stress Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "plot10_social_media_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plot10_social_media_by_class.png")


def save_summary_csvs(
    *,
    output_dir: Path,
    cv_scores: np.ndarray,
    val_metrics: Dict[str, float | np.ndarray],
    test_metrics: Dict[str, float | np.ndarray],
    perm_df: pd.DataFrame,
) -> None:
    results_dict = {
        "Metric": [
            "CV Accuracy",
            "CV Std",
            "Val Accuracy",
            "Val F1",
            "Val AUC",
            "Test Accuracy",
            "Test F1",
            "Test AUC",
        ],
        "Value": [
            float(cv_scores.mean()),
            float(cv_scores.std()),
            float(val_metrics["accuracy"]),
            float(val_metrics["f1_weighted"]),
            float(val_metrics["auc_ovr_weighted"]),
            float(test_metrics["accuracy"]),
            float(test_metrics["f1_weighted"]),
            float(test_metrics["auc_ovr_weighted"]),
        ],
    }

    pd.DataFrame(results_dict).to_csv(output_dir / "model_results.csv", index=False)
    perm_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print("Saved: model_results.csv")
    print("Saved: feature_importance.csv")


def main() -> None:
    # STEP 1 — Load datasets
    print_step("STEP 1 — LOADING DATASETS")
    df1, df2_raw, df3 = load_datasets(DATA_DIR)
    print(f"DS1 Mobile Usage:        {df1.shape}")
    print(f"DS2 Social Media & MH:   {df2_raw.shape}")
    print(f"DS3 Remote Work & MH:    {df3.shape}")

    # STEP 2 — Preprocess DS2 (Primary)
    print_step("STEP 2 — PREPROCESSING DS2 (PRIMARY)")
    df2, t33, t66 = preprocess_primary_dataset(df2_raw)
    print(f"DS2 cleaned shape: {df2.shape}")
    print(f"Target tertile thresholds: Low <= {t33:.2f}, Medium <= {t66:.2f}, High > {t66:.2f}")
    print_class_distribution(df2["Stress_Label"])

    # STEP 3 — Enrichment from DS1 & DS3
    print_step("STEP 3 — PREPROCESSING DS1 & DS3 FOR ENRICHMENT")
    ds1_enriched = build_ds1_enrichment(df1)
    ds3_enriched = build_ds3_enrichment(df3)
    print("DS1 Age-group enrichment:")
    print(ds1_enriched.to_string(index=False))
    print("\nDS3 Age-group enrichment:")
    print(ds3_enriched.to_string(index=False))

    # STEP 4 — Merge into unified dataset
    print_step("STEP 4 — MERGING INTO UNIFIED CROSS-DEVICE DATASET")
    df_main = merge_to_unified_dataset(df2, ds1_enriched, ds3_enriched)
    print(f"Unified dataset: {df_main.shape}")
    final_dist = (
        df_main["Stress_Label"]
        .value_counts()
        .sort_index()
        .rename({0: "Low", 1: "Medium", 2: "High"})
    )
    print("Final class distribution:")
    print(final_dist)

    # STEP 5 — Feature selection & normalisation
    print_step("STEP 5 — FEATURE SELECTION & NORMALISATION")
    x_scaled, y, feature_cols = prepare_features_and_target(df_main)
    print(f"Feature matrix: {x_scaled.shape}")

    # STEP 6 — Split (70/15/15)
    print_step("STEP 6 — TRAIN / VALIDATION / TEST SPLIT")
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x_scaled, y, random_state=42
    )
    print(f"Train:      {x_train.shape[0]} samples")
    print(f"Validation: {x_val.shape[0]} samples")
    print(f"Test:       {x_test.shape[0]} samples")

    # STEP 7 — Train model + CV
    print_step("STEP 7 — GAUSSIAN NAIVE BAYES TRAINING")
    gnb, cv_scores = train_gnb(x_train, y_train, random_state=42)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    val_metrics = evaluate_split(gnb, x_val, y_val)
    print(f"Validation Accuracy:      {val_metrics['accuracy']:.4f}")
    print(f"Validation F1 (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Validation AUC (OvR):     {val_metrics['auc_ovr_weighted']:.4f}")

    # STEP 8 — Test evaluation
    print_step("STEP 8 — TEST SET EVALUATION")
    test_metrics = evaluate_split(gnb, x_test, y_test)
    print(f"Test Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted):       {test_metrics['f1_weighted']:.4f}")
    print(f"Test AUC (OvR weighted):  {test_metrics['auc_ovr_weighted']:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_metrics["y_pred"],
            target_names=CLASS_NAMES,
        )
    )

    # XAI — Permutation importance
    perm_df = compute_permutation_importance(gnb, x_test, y_test, feature_cols, random_state=42)
    print("\nTop 15 Features — Permutation Importance:")
    print(perm_df.head(15).to_string(index=False))

    # STEP 9 — Plots
    generate_all_plots(
        output_dir=OUTPUT_DIR,
        df_main=df_main,
        y_all=y,
        feature_cols=feature_cols,
        cv_scores=cv_scores,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        y_test=y_test,
        y_test_pred=test_metrics["y_pred"],
        y_test_proba=test_metrics["y_proba"],
        perm_df=perm_df,
        model=gnb,
        x_test=x_test,
    )

    # Final summary
    print_step("FINAL RESULTS SUMMARY")
    print(f"Total samples (final):    {len(x_scaled)}")
    print(f"Total features:           {len(feature_cols)}")
    print("  - DS2 (social/MH):      12 direct + 4 demographic")
    print("  - DS1 (mobile usage):   6 enriched features")
    print("  - DS3 (work/PC):        5 enriched features")
    print(
        f"Train / Val / Test:       {x_train.shape[0]} / {x_val.shape[0]} / {x_test.shape[0]}"
    )
    print("\n--- Performance ---")
    print(f"CV Accuracy (5-fold):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Validation Accuracy:      {val_metrics['accuracy']:.4f}")
    print(f"Validation F1 (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Validation AUC (OvR):     {val_metrics['auc_ovr_weighted']:.4f}")
    print(f"Test Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted):       {test_metrics['f1_weighted']:.4f}")
    print(f"Test AUC (OvR):           {test_metrics['auc_ovr_weighted']:.4f}")
    print("\n--- Top 5 Most Influential Features (XAI) ---")
    for _, row in perm_df.head(5).iterrows():
        print(f"  {row['Feature']:<32} Importance: {row['Importance']:.4f}")
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    # Save results to CSV for reference
    save_summary_csvs(
        output_dir=OUTPUT_DIR,
        cv_scores=cv_scores,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        perm_df=perm_df,
    )
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
