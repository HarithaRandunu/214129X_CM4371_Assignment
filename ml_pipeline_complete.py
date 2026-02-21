# Digital Wellbeing Stress Predictor — ML Pipeline
# Student: 214129X — Malalpola MLHR
#
# Reads three CSV datasets from data_set/, trains a Gaussian Naive Bayes model,
# evaluates on a held-out test split, runs XAI analysis, and saves CSV outputs.
#
# Run: python ml_pipeline_complete.py

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

from project_config import (
    CLASS_NAMES,
    DATASET_LINKS,
    DATA_DIR,
    DS2_COL_MAP,
    FEATURE_COLS,
    MODEL_CONFIG,
    OUTPUT_DIR,
    OUTPUT_FILES,
    SM_HOURS_MAP,
    STRESS_INDICATOR_COLS,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def age_group(age: float) -> str:
    if age < 25: return "18-24"
    if age < 35: return "25-34"
    if age < 45: return "35-44"
    if age < 55: return "45-54"
    return "55+"


def classify_stress(score: float, low_threshold: float, high_threshold: float) -> int:
    if score <= low_threshold:
        return 0
    if score <= high_threshold:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Step 1: Load raw datasets
# ---------------------------------------------------------------------------

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        DATA_DIR / "user_behavior_dataset.csv",
        DATA_DIR / "smmh.csv",
        DATA_DIR / "Impact_of_Remote_Work_on_Mental_Health.csv",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))

    df1, df2, df3 = [pd.read_csv(p) for p in paths]
    return df1, df2, df3


# ---------------------------------------------------------------------------
# Step 2: Preprocess DS2 — rename, encode, engineer stress label
# ---------------------------------------------------------------------------

def preprocess_ds2(df2_raw: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    df = df2_raw.rename(columns=DS2_COL_MAP).copy()
    df["Social_Media_Hours"] = df["Social_Media_Hours"].map(SM_HOURS_MAP)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age", "Social_Media_Hours"]).copy()

    le = LabelEncoder()
    for col in ("Gender", "Relationship_Status", "Occupation"):
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str).str.strip())

    df["Composite_Stress"] = df[STRESS_INDICATOR_COLS].mean(axis=1)
    t33 = float(df["Composite_Stress"].quantile(0.33))
    t66 = float(df["Composite_Stress"].quantile(0.66))
    df["Stress_Label"] = df["Composite_Stress"].apply(
        lambda score: classify_stress(score, t33, t66)
    )
    return df, t33, t66


# ---------------------------------------------------------------------------
# Step 3: Aggregate DS1 (mobile usage) and DS3 (work/PC) by age group
# ---------------------------------------------------------------------------

def aggregate_ds1(df1: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df1["Age_Group"] = df1["Age"].apply(age_group)
    return df1.groupby("Age_Group").agg(
        Avg_Screen_Time_hrs   = ("Screen On Time (hours/day)",  "mean"),
        Avg_Battery_Drain_mAh = ("Battery Drain (mAh/day)",     "mean"),
        Avg_App_Usage_min     = ("App Usage Time (min/day)",    "mean"),
        Avg_Data_Usage_MB     = ("Data Usage (MB/day)",         "mean"),
        Avg_Apps_Installed    = ("Number of Apps Installed",    "mean"),
        Avg_Mobile_Intensity  = ("User Behavior Class",         "mean"),
    ).reset_index()


def aggregate_ds3(df3: pd.DataFrame) -> pd.DataFrame:
    df3 = df3.copy()
    df3["Physical_Activity"] = df3["Physical_Activity"].fillna(
        df3["Physical_Activity"].mode()[0]
    )
    df3 = df3.dropna(subset=["Mental_Health_Condition"]).copy()
    df3["Age_Group"] = df3["Age"].apply(age_group)
    df3["Sleep_Quality_num"] = df3["Sleep_Quality"].map({"Poor": 1, "Average": 2, "Good": 3})
    return df3.groupby("Age_Group").agg(
        Avg_Work_Hours_Week   = ("Hours_Worked_Per_Week",     "mean"),
        Avg_Work_Life_Balance = ("Work_Life_Balance_Rating",  "mean"),
        Avg_Social_Isolation  = ("Social_Isolation_Rating",   "mean"),
        Avg_Sleep_Quality     = ("Sleep_Quality_num",         "mean"),
        Avg_Virtual_Meetings  = ("Number_of_Virtual_Meetings","mean"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Step 4: Merge into unified cross-device dataset
# ---------------------------------------------------------------------------

def merge_datasets(
    df2: pd.DataFrame,
    ds1: pd.DataFrame,
    ds3: pd.DataFrame,
) -> pd.DataFrame:
    df2 = df2.copy()
    df2["Age_Group"] = df2["Age"].apply(age_group)
    merged = (
        df2
        .merge(ds1, on="Age_Group", how="left", validate="many_to_one")
        .merge(ds3, on="Age_Group", how="left", validate="many_to_one")
        .dropna()
        .copy()
    )
    return merged


# ---------------------------------------------------------------------------
# Step 5: Scale features and split into train / val / test
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    x = df[FEATURE_COLS].copy()
    y = df["Stress_Label"].copy()
    scaler = StandardScaler()
    x_sc = pd.DataFrame(scaler.fit_transform(x), columns=FEATURE_COLS)
    return x_sc, y, scaler


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = MODEL_CONFIG["random_state"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x,
        y,
        test_size=MODEL_CONFIG["train_split_test_size"],
        random_state=random_state,
        stratify=y,
    )
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp,
        y_tmp,
        test_size=MODEL_CONFIG["val_test_split_size"],
        random_state=random_state,
        stratify=y_tmp,
    )
    return x_tr, x_val, x_te, y_tr, y_val, y_te


# ---------------------------------------------------------------------------
# Step 6: Train Gaussian Naive Bayes with stratified cross-validation
# ---------------------------------------------------------------------------

def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = MODEL_CONFIG["random_state"],
) -> Tuple[GaussianNB, np.ndarray]:
    model = GaussianNB()
    model.fit(x_train, y_train)
    cv = StratifiedKFold(
        n_splits=MODEL_CONFIG["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    return model, cv_scores


# ---------------------------------------------------------------------------
# Step 7: Evaluate on a given split and return metrics
# ---------------------------------------------------------------------------

def evaluate(
    model: GaussianNB,
    x: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, object]:
    y_pred  = model.predict(x)
    y_proba = model.predict_proba(x)
    return {
        "y_pred":            y_pred,
        "y_proba":           y_proba,
        "accuracy":          float(accuracy_score(y, y_pred)),
        "f1_weighted":       float(f1_score(y, y_pred, average="weighted")),
        "auc_ovr_weighted":  float(roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")),
    }


# ---------------------------------------------------------------------------
# Step 8: Permutation feature importance (XAI method 1)
# ---------------------------------------------------------------------------

def permutation_importance_df(
    model: GaussianNB,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = MODEL_CONFIG["random_state"],
) -> pd.DataFrame:
    result = permutation_importance(
        model, x_test, y_test,
        n_repeats=MODEL_CONFIG["perm_repeats"],
        random_state=random_state,
        scoring="accuracy",
    )
    return (
        pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": result.importances_mean,
            "Std":        result.importances_std,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Step 10: Save result CSVs
# ---------------------------------------------------------------------------

def save_csvs(
    *,
    cv_scores: np.ndarray,
    val_metrics: Dict,
    test_metrics: Dict,
    perm_df: pd.DataFrame,
    out: Path,
) -> None:
    pd.DataFrame({
        "Metric": [
            "CV Accuracy", "CV Std",
            "Val Accuracy", "Val F1", "Val AUC",
            "Test Accuracy", "Test F1", "Test AUC",
        ],
        "Value": [
            cv_scores.mean(), cv_scores.std(),
            val_metrics["accuracy"], val_metrics["f1_weighted"], val_metrics["auc_ovr_weighted"],
            test_metrics["accuracy"], test_metrics["f1_weighted"], test_metrics["auc_ovr_weighted"],
        ],
    }).to_csv(out / OUTPUT_FILES["model_results"], index=False)

    perm_df.to_csv(out / OUTPUT_FILES["feature_importance"], index=False)
    print(
        f"Saved: {OUTPUT_FILES['model_results']}, {OUTPUT_FILES['feature_importance']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    section("STEP 1 — LOADING DATASETS")
    df1, df2_raw, df3 = load_datasets()
    print(f"DS1 shape: {df1.shape}  |  DS2 shape: {df2_raw.shape}  |  DS3 shape: {df3.shape}")
    print("Dataset source pages:")
    for name, url in DATASET_LINKS.items():
        print(f"  {name}: {url}")

    section("STEP 2 — PREPROCESSING DS2 (PRIMARY)")
    df2, t33, t66 = preprocess_ds2(df2_raw)
    print(f"DS2 cleaned: {df2.shape}")
    print(f"Stress thresholds — Low <= {t33:.2f}  |  Medium <= {t66:.2f}  |  High > {t66:.2f}")
    print(f"Class distribution: {dict(df2['Stress_Label'].value_counts().sort_index())}")

    section("STEP 3 — AGGREGATING DS1 AND DS3 BY AGE GROUP")
    ds1_agg = aggregate_ds1(df1)
    ds3_agg = aggregate_ds3(df3)
    print("DS1 age-group enrichment:\n", ds1_agg.to_string(index=False))
    print("\nDS3 age-group enrichment:\n", ds3_agg.to_string(index=False))

    section("STEP 4 — MERGING INTO UNIFIED DATASET")
    df_main = merge_datasets(df2, ds1_agg, ds3_agg)
    print(f"Unified dataset: {df_main.shape}")
    print(f"Class distribution: {dict(df_main['Stress_Label'].value_counts().sort_index())}")

    section("STEP 5 — FEATURE SCALING AND SPLITTING")
    x_sc, y, _ = prepare_features(df_main)
    x_tr, x_val, x_te, y_tr, y_val, y_te = split_data(x_sc, y)
    print(f"Train: {len(x_tr)}  |  Val: {len(x_val)}  |  Test: {len(x_te)}")

    section("STEP 6 — TRAINING GAUSSIAN NAIVE BAYES")
    model, cv_scores = train_model(x_tr, y_tr)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    section("STEP 7 — VALIDATION EVALUATION")
    val_m = evaluate(model, x_val, y_val)
    print(f"Val Accuracy: {val_m['accuracy']:.4f}  |  F1: {val_m['f1_weighted']:.4f}  |  AUC: {val_m['auc_ovr_weighted']:.4f}")

    section("STEP 8 — TEST SET EVALUATION")
    test_m = evaluate(model, x_te, y_te)
    print(f"Test Accuracy: {test_m['accuracy']:.4f}  |  F1: {test_m['f1_weighted']:.4f}  |  AUC: {test_m['auc_ovr_weighted']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te, test_m["y_pred"], target_names=CLASS_NAMES))

    section("STEP 9 — XAI: PERMUTATION FEATURE IMPORTANCE")
    perm_df = permutation_importance_df(model, x_te, y_te)
    print("Top 10 features:")
    print(perm_df.head(10).to_string(index=False))

    section("STEP 10 — SAVING OUTPUT CSV FILES")
    save_csvs(cv_scores=cv_scores, val_metrics=val_m,
              test_metrics=test_m, perm_df=perm_df, out=OUTPUT_DIR)

    section("FINAL SUMMARY")
    print(f"Samples: {len(x_sc)}  |  Features: {len(FEATURE_COLS)}")
    print(f"Split — Train: {len(x_tr)} / Val: {len(x_val)} / Test: {len(x_te)}")
    print(f"CV Accuracy:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Test Accuracy:  {test_m['accuracy']:.4f}")
    print(f"Test F1:        {test_m['f1_weighted']:.4f}")
    print(f"Test AUC:       {test_m['auc_ovr_weighted']:.4f}")
    print("\nTop 5 features by importance:")
    for _, row in perm_df.head(5).iterrows():
        print(f"  {row['Feature']:<32}  {row['Importance']:.4f}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()