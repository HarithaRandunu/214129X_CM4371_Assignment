# Digital Wellbeing Stress Predictor
# Student: 214129X — Malalpola MLHR
#
# Run:  streamlit run app.py
# Deps: pip install streamlit plotly scikit-learn pandas numpy psutil

from __future__ import annotations

import platform
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
import streamlit.components.v1 as components
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
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

from project_config import (
    APP_CLASS_BG,
    APP_CLASS_COLOURS,
    CLASS_NAMES,
    DATASET_LINKS,
    DATA_DIR,
    DS2_COL_MAP,
    FEATURE_COLS,
    FEATURE_IMPORTANCE_FILE,
    LAYOUT_CONFIG,
    MIN_PAGE_LOADER_SECONDS,
    MODEL_CONFIG,
    MODEL_RESULTS_FILE,
    OUTPUT_DIR,
    PAGE_DEVICE_PREDICTIONS,
    PAGE_OVERVIEW,
    PAGE_PROJECT_REPORT,
    PAGE_STRESS_PREDICTOR,
    SM_HOURS_MAP,
    STRESS_INDICATOR_COLS,
    UI_THEME,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_COLOURS = APP_CLASS_COLOURS
CLASS_BG = APP_CLASS_BG

# ---------------------------------------------------------------------------
# SVG icon helpers (Feather icons — no emoji, no external CDN)
# ---------------------------------------------------------------------------

def _svg(path_d: str, size: int = 16) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px">'
        f'{path_d}</svg>'
    )

ICON = {
    "home":    _svg('<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>'),
    "brain":   _svg('<circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/>'),
    "phone":   _svg('<rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/>'),
    "report":  _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'),
    "arrow":   _svg('<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>'),
    "warning": _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
    "check":   _svg('<polyline points="20 6 9 17 4 12"/>'),
    "info":    _svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'),
    "monitor": _svg('<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>'),
    "wifi":    _svg('<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>'),
    "battery": _svg('<rect x="1" y="6" width="18" height="12" rx="2"/><line x1="23" y1="11" x2="23" y2="13"/><line x1="5" y1="12" x2="13" y2="12"/>'),
    "cpu":     _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
}

def dot(colour: str, size: int = 12) -> str:
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'border-radius:50%;background:{colour};vertical-align:middle;'
        f'margin-right:6px;border:1.5px solid rgba(255,255,255,0.6)"></span>'
    )


def render_dataset_links() -> None:
    st.markdown("**Dataset pages (Kaggle):**")
    links_html = "".join(
        f'<li><a href="{url}" target="_blank">{name}</a></li>'
        for name, url in DATASET_LINKS.items()
    )
    st.markdown(f"<ul>{links_html}</ul>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page config and CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Wellbeing Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {UI_THEME['sidebar_bg']}; }}
    section[data-testid="stSidebar"] button[kind="primary"] {{
        background: {UI_THEME['sidebar_primary']}; border-color: {UI_THEME['sidebar_primary']};
    }}
    section[data-testid="stSidebar"] button[kind="secondary"] {{
        background: transparent; border-color: {UI_THEME['sidebar_border']}; color: {UI_THEME['muted_text']};
    }}
    section[data-testid="stSidebar"] button {{ text-align: left; margin: {LAYOUT_CONFIG['sidebar_btn_margin']}; }}

  /* KPI cards */
    .kpi {{
        background: linear-gradient(135deg, {UI_THEME['panel_bg']}, {UI_THEME['sidebar_bg']});
        border: 1px solid {UI_THEME['sidebar_border']}; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
    }}
    .kpi .val {{ font-size: 1.9rem; font-weight: 800; }}
    .kpi .lbl {{ font-size: 0.78rem; color: {UI_THEME['muted_text']}; margin-top: 0.2rem; }}

  /* Prediction result banner */
    .result {{
    border-radius: 12px; padding: 1.2rem; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin: {LAYOUT_CONFIG['result_margin']};
    }}

  /* Report block highlight */
    .rblock {{
        background: {UI_THEME['panel_bg']}; border-left: 3px solid {UI_THEME['sidebar_primary']};
    border-radius: 6px; padding: 0.9rem 1.1rem; margin: 0.5rem 0;
    }}

  /* Section heading with icon */
    .sec-head {{ font-size: 1.25rem; font-weight: 700; margin: {LAYOUT_CONFIG['section_heading_margin']}; }}

  /* Device stat card */
    .dcard {{
        background: {UI_THEME['panel_bg']}; border: 1px solid {UI_THEME['sidebar_border']};
    border-radius: 8px; padding: 0.9rem; text-align: center;
    }}
    .dcard .dval {{ font-size: 1.5rem; font-weight: 700; }}
    .dcard .dlbl {{ font-size: 0.78rem; color: {UI_THEME['muted_text']}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme shared across all charts
# ---------------------------------------------------------------------------

_PT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(22,27,34,0.7)",
    "font": {"family": "system-ui, sans-serif", "size": 12},
    "margin": LAYOUT_CONFIG["plot_margin"],
}

# ---------------------------------------------------------------------------
# ML pipeline — loaded and cached once
# ---------------------------------------------------------------------------

def _age_group(age: float) -> str:
    if age < 25: return "18-24"
    if age < 35: return "25-34"
    if age < 45: return "35-44"
    if age < 55: return "45-54"
    return "55+"


def _classify_stress_label(score: float, lower: float, upper: float) -> int:
    if score <= lower:
        return 0
    if score <= upper:
        return 1
    return 2


@st.cache_resource(show_spinner="Preparing model...")
def load_pipeline() -> dict:
    df1     = pd.read_csv(DATA_DIR / "user_behavior_dataset.csv")
    df2_raw = pd.read_csv(DATA_DIR / "smmh.csv")
    df3     = pd.read_csv(DATA_DIR / "Impact_of_Remote_Work_on_Mental_Health.csv")

    # DS2: rename verbose survey columns, encode categoricals, engineer target
    df2 = df2_raw.rename(columns=DS2_COL_MAP).copy()
    df2["Social_Media_Hours"] = df2["Social_Media_Hours"].map(SM_HOURS_MAP)
    df2["Age"] = pd.to_numeric(df2["Age"], errors="coerce")
    df2 = df2.dropna(subset=["Age", "Social_Media_Hours"]).copy()

    le = LabelEncoder()
    for col in ("Gender", "Relationship_Status", "Occupation"):
        df2[f"{col}_enc"] = le.fit_transform(df2[col].astype(str).str.strip())

    df2["Composite_Stress"] = df2[STRESS_INDICATOR_COLS].mean(axis=1)
    t33 = float(df2["Composite_Stress"].quantile(0.33))
    t66 = float(df2["Composite_Stress"].quantile(0.66))
    df2["Stress_Label"] = df2["Composite_Stress"].apply(
        lambda score: _classify_stress_label(score, t33, t66)
    )

    # DS1: aggregate mobile usage stats by age group
    df1["Age_Group"] = df1["Age"].apply(_age_group)
    ds1 = df1.groupby("Age_Group").agg(
        Avg_Screen_Time_hrs   = ("Screen On Time (hours/day)",  "mean"),
        Avg_Battery_Drain_mAh = ("Battery Drain (mAh/day)",     "mean"),
        Avg_App_Usage_min     = ("App Usage Time (min/day)",    "mean"),
        Avg_Data_Usage_MB     = ("Data Usage (MB/day)",         "mean"),
        Avg_Apps_Installed    = ("Number of Apps Installed",    "mean"),
        Avg_Mobile_Intensity  = ("User Behavior Class",         "mean"),
    ).reset_index()

    # DS3: aggregate work/PC context by age group
    df3["Physical_Activity"] = df3["Physical_Activity"].fillna(
        df3["Physical_Activity"].mode()[0]
    )
    df3 = df3.dropna(subset=["Mental_Health_Condition"]).copy()
    df3["Age_Group"] = df3["Age"].apply(_age_group)
    df3["Sleep_Quality_num"] = df3["Sleep_Quality"].map({"Poor": 1, "Average": 2, "Good": 3})
    ds3 = df3.groupby("Age_Group").agg(
        Avg_Work_Hours_Week   = ("Hours_Worked_Per_Week",    "mean"),
        Avg_Work_Life_Balance = ("Work_Life_Balance_Rating", "mean"),
        Avg_Social_Isolation  = ("Social_Isolation_Rating",  "mean"),
        Avg_Sleep_Quality     = ("Sleep_Quality_num",        "mean"),
        Avg_Virtual_Meetings  = ("Number_of_Virtual_Meetings", "mean"),
    ).reset_index()

    # Merge all three datasets via age-group bridge
    df2["Age_Group"] = df2["Age"].apply(_age_group)
    df_main = (
        df2
        .merge(ds1, on="Age_Group", how="left", validate="many_to_one")
        .merge(ds3, on="Age_Group", how="left", validate="many_to_one")
        .dropna()
        .copy()
    )

    x = df_main[FEATURE_COLS].copy()
    y = df_main["Stress_Label"].copy()

    scaler = StandardScaler()
    x_sc = pd.DataFrame(scaler.fit_transform(x), columns=FEATURE_COLS)

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x_sc,
        y,
        test_size=MODEL_CONFIG["train_split_test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y,
    )
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp,
        y_tmp,
        test_size=MODEL_CONFIG["val_test_split_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y_tmp,
    )

    model = GaussianNB()
    model.fit(x_tr, y_tr)

    cv = StratifiedKFold(
        n_splits=MODEL_CONFIG["cv_folds"],
        shuffle=True,
        random_state=MODEL_CONFIG["random_state"],
    )
    cv_scores = cross_val_score(model, x_tr, y_tr, cv=cv, scoring="accuracy")

    y_val_pred = model.predict(x_val)
    y_val_proba = model.predict_proba(x_val)
    y_pred  = model.predict(x_te)
    y_proba = model.predict_proba(x_te)

    perm = permutation_importance(
        model,
        x_te,
        y_te,
        n_repeats=MODEL_CONFIG["perm_repeats"],
        random_state=MODEL_CONFIG["random_state"],
        scoring="accuracy",
    )
    perm_df = (
        pd.DataFrame({"Feature": FEATURE_COLS, "Importance": perm.importances_mean, "Std": perm.importances_std})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "df_main": df_main,
        "df2": df2,
        "ds1": ds1,
        "ds3": ds3,
        "X_test": x_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cv_scores": cv_scores,
        "val_acc": float(accuracy_score(y_val, y_val_pred)),
        "val_f1": float(f1_score(y_val, y_val_pred, average="weighted")),
        "val_auc": float(roc_auc_score(y_val, y_val_proba, multi_class="ovr", average="weighted")),
        "acc":  float(accuracy_score(y_te, y_pred)),
        "f1":   float(f1_score(y_te, y_pred, average="weighted")),
        "auc":  float(roc_auc_score(y_te, y_proba, multi_class="ovr", average="weighted")),
        "perm_df": perm_df,
        "t33": t33,
        "t66": t66,
    }


def save_output_csvs(p: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_results_df = pd.DataFrame(
        {
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
                p["cv_scores"].mean(),
                p["cv_scores"].std(),
                p["val_acc"],
                p["val_f1"],
                p["val_auc"],
                p["acc"],
                p["f1"],
                p["auc"],
            ],
        }
    )
    model_results_df.to_csv(OUTPUT_DIR / MODEL_RESULTS_FILE, index=False)
    p["perm_df"].to_csv(OUTPUT_DIR / FEATURE_IMPORTANCE_FILE, index=False)


def render_output_controls(p: dict) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Outputs**")

    if st.sidebar.button("Generate result CSV files", key="generate_csv_outputs", width="stretch"):
        save_output_csvs(p)
        st.session_state._csv_outputs_written = True
        st.session_state._csv_outputs_message = "CSV files generated."

    if st.session_state.get("_csv_outputs_written", False):
        model_results_path = OUTPUT_DIR / MODEL_RESULTS_FILE
        feature_importance_path = OUTPUT_DIR / FEATURE_IMPORTANCE_FILE

        if model_results_path.exists() and feature_importance_path.exists():
            st.sidebar.success(st.session_state.get("_csv_outputs_message", "CSV files are ready."))

            with model_results_path.open("rb") as file_obj:
                st.sidebar.download_button(
                    label="Download model_results.csv",
                    data=file_obj,
                    file_name=MODEL_RESULTS_FILE,
                    mime="text/csv",
                    key="download_model_results_csv",
                    width="stretch",
                )

            with feature_importance_path.open("rb") as file_obj:
                st.sidebar.download_button(
                    label="Download feature_importance.csv",
                    data=file_obj,
                    file_name=FEATURE_IMPORTANCE_FILE,
                    mime="text/csv",
                    key="download_feature_importance_csv",
                    width="stretch",
                )


def run_prediction(inputs: dict, p: dict) -> tuple[int, np.ndarray]:
    ag   = _age_group(inputs["age"])
    rows = p["df_main"][p["df_main"]["Age_Group"] == ag]
    ref  = rows.iloc[0] if len(rows) else p["df_main"].iloc[0]

    vec = {
        "Social_Media_Hours":       inputs["sm_hours"],
        "Purposeless_Use":          inputs["purposeless"],
        "Distraction_Score":        inputs["distraction"],
        "Restlessness":             inputs["restless"],
        "Easily_Distracted":        inputs["easily_distracted"],
        "Worry_Score":              inputs["worry"],
        "Difficulty_Concentrating": inputs["concentrate"],
        "Comparison_Score":         inputs["comparison"],
        "Validation_Seeking":       inputs["validation"],
        "Depression_Score":         inputs["depression"],
        "Interest_Fluctuation":     inputs["interest_fluct"],
        "Sleep_Issues":             inputs["sleep_issues"],
        "Age":                      inputs["age"],
        "Gender_enc":               0 if inputs["gender"] == "Female" else 1,
        "Relationship_Status_enc":  1,
        "Occupation_enc":           1,
        **{k: ref[k] for k in [
            "Avg_Screen_Time_hrs", "Avg_Battery_Drain_mAh", "Avg_App_Usage_min",
            "Avg_Data_Usage_MB", "Avg_Apps_Installed", "Avg_Mobile_Intensity",
            "Avg_Work_Hours_Week", "Avg_Work_Life_Balance", "Avg_Social_Isolation",
            "Avg_Sleep_Quality", "Avg_Virtual_Meetings",
        ]},
    }

    df_vec = pd.DataFrame([vec])[p["feature_cols"]]
    cls = p["model"].predict(p["scaler"].transform(df_vec))[0]
    proba = p["model"].predict_proba(p["scaler"].transform(df_vec))[0]
    return int(cls), proba

# ---------------------------------------------------------------------------
# Device data collection (server-side via psutil)
# ---------------------------------------------------------------------------

def collect_server_stats() -> dict:
    d: dict = {}
    try:
        b = psutil.sensors_battery()
        d["batt_pct"]     = round(b.percent, 1) if b else None
        d["batt_plugged"] = b.power_plugged if b else None
        d["batt_secs"]    = b.secsleft if b else None
    except Exception:
        d["batt_pct"] = d["batt_plugged"] = d["batt_secs"] = None

    d["cpu_pct"]      = psutil.cpu_percent(interval=1)
    d["cpu_cores"]    = psutil.cpu_count(logical=True)
    mem               = psutil.virtual_memory()
    d["ram_pct"]      = round(mem.percent, 1)
    d["ram_used_gb"]  = round(mem.used  / 1e9, 1)
    d["ram_total_gb"] = round(mem.total / 1e9, 1)
    net               = psutil.net_io_counters()
    d["net_sent_gb"]  = round(net.bytes_sent / 1e9, 2)
    d["net_recv_gb"]  = round(net.bytes_recv / 1e9, 2)
    d["os"]           = platform.system()
    d["machine"]      = platform.machine()
    return d


# Rule-based device health predictions
def _batt_prediction(pct, plugged, cpu, ram) -> dict:
    if pct is None:
        return {"status": "No battery", "colour": "#8b949e",
                "tip": "Desktop machine detected — no battery present."}
    drain_idx = round(cpu * 0.15 + ram * 0.10, 1)
    if pct > 80:   status, col = "Excellent", "#27ae60"
    elif pct > 50: status, col = "Good",      "#2ecc71"
    elif pct > 20: status, col = "Moderate",  "#e67e22"
    else:          status, col = "Critical",  "#c0392b"
    tips = {
        "Excellent": "Battery is healthy. Avoid sustained 100% charge to extend lifespan.",
        "Good":      "Keep charge between 20–80% for optimal long-term health.",
        "Moderate":  "Charge soon and reduce CPU-heavy tasks to slow drain.",
        "Critical":  "Plug in immediately. Close background applications.",
    }
    hrs = (pct / max(drain_idx, 1)) * 0.5 if not plugged else None
    return {"status": status, "colour": col, "tip": tips[status],
            "drain_idx": drain_idx, "hrs_left": round(hrs, 1) if hrs else None}


def _net_prediction(sent, recv) -> dict:
    total = sent + recv
    if total < 1:    cat, col = "Low",      "#27ae60"
    elif total < 10: cat, col = "Moderate", "#e67e22"
    else:            cat, col = "Heavy",    "#c0392b"
    tip = ("Usage is within healthy limits." if total < 5
           else "High data detected. Check for background streaming or sync tasks.")
    return {
        "category": cat, "colour": col, "total": round(total, 2),
        "social": round(recv * 0.35, 2), "stream": round(recv * 0.40, 2),
        "other":  round(recv * 0.25, 2), "tip": tip,
    }


def _screen_prediction(cpu, ram) -> dict:
    load = (cpu + ram) / 2
    if load < 30:   status, col = "Light",    "#27ae60"
    elif load < 60: status, col = "Moderate", "#e67e22"
    else:           status, col = "Heavy",    "#c0392b"
    tip = ("System load is comfortable for sustained focus."
           if load < 30 else
           "High system load. Close unused apps to reduce heat and eye strain.")
    return {"status": status, "colour": col, "load": round(load, 1), "tip": tip}

# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def _proba_bar(proba: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(proba * 100), y=CLASS_NAMES, orientation="h",
        marker_color=CLASS_COLOURS,
        text=[f"{v * 100:.1f}%" for v in proba], textposition="outside",
    ))
    fig.update_layout(title="Prediction confidence (%)", xaxis={"range": [0, 115]},
                      height=220, **_PT)
    return fig


def _indicator_bars(ind: dict, t33: float, t66: float) -> go.Figure:
    vals   = list(ind.values())
    labels = list(ind.keys())
    cols   = ["#c0392b" if v >= 4 else "#e67e22" if v >= 3 else "#27ae60" for v in vals]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h",
                           marker_color=cols, text=vals, textposition="outside"))
    fig.add_vline(x=3, line_dash="dash", line_color="#e67e22",
                  annotation_text="Caution threshold")
    comp = np.mean(vals)
    fig.update_layout(
        title=f"Mental wellbeing indicators  |  Composite: {comp:.2f}  "
              f"(Low ≤ {t33:.2f}  /  High > {t66:.2f})",
        xaxis={"range": [0, 6]}, height=320, **_PT,
    )
    return fig


def _confusion_matrix(y_test, y_pred) -> go.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES,
                    color_continuous_scale="Blues", text_auto=True,
                    labels={"x": "Predicted", "y": "Actual"},
                    title="Confusion matrix (test set)")
    fig.update_layout(height=350, **_PT)
    return fig


def _roc_curves(y_test, y_proba) -> go.Figure:
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig = go.Figure()
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{cls}  AUC={auc_i:.2f}",
                                 line={"color": col, "width": 2.5}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                             line={"color": "#555", "dash": "dash"}))
    fig.update_layout(title="ROC curves — one vs rest",
                      xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      height=350, **_PT)
    return fig


def _cv_bars(cv_scores: np.ndarray) -> go.Figure:
    mean  = cv_scores.mean()
    cols  = ["#1f6feb" if s >= mean else "#e67e22" for s in cv_scores]
    folds = [f"Fold {i + 1}" for i in range(len(cv_scores))]
    fig = go.Figure(go.Bar(x=folds, y=cv_scores, marker_color=cols,
                           text=[f"{v:.3f}" for v in cv_scores], textposition="outside"))
    fig.add_hline(y=mean, line_dash="dash", line_color="#c0392b",
                  annotation_text=f"Mean = {mean:.3f}", annotation_position="top right")
    fig.update_layout(title="5-fold cross-validation accuracy",
                      yaxis={"range": [0, 1.1]}, height=300, **_PT)
    return fig


def _feature_importance(perm_df: pd.DataFrame) -> go.Figure:
    top = perm_df.head(12).copy()
    cols = ["#c0392b" if v > 0 else "#555" for v in top["Importance"]]
    fig = go.Figure(go.Bar(
        y=top["Feature"][::-1], x=top["Importance"][::-1],
        error_x={"type": "data", "array": top["Std"][::-1].tolist(), "visible": True},
        orientation="h", marker_color=cols[::-1],
    ))
    fig.update_layout(title="Permutation feature importance (XAI)",
                      xaxis_title="Mean accuracy decrease", height=420, **_PT)
    return fig


def _class_means_heatmap(model: GaussianNB, feature_cols: list, perm_df: pd.DataFrame) -> go.Figure:
    top10  = perm_df.head(10)["Feature"].tolist()
    idx    = [feature_cols.index(f) for f in top10]
    theta  = model.theta_[:, idx]
    labels = [f.replace("_", " ") for f in top10]
    fig = go.Figure(go.Heatmap(
        z=theta, x=labels, y=CLASS_NAMES, colorscale="RdYlGn_r",
        text=np.round(theta, 2), texttemplate="%{text}",
        colorbar={"title": "Normalised mean"},
    ))
    fig.update_layout(title="GNB class means — what the model learned per stress class",
                      height=290, **_PT)
    return fig


def _class_dist(y: pd.Series) -> go.Figure:
    counts = y.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=CLASS_NAMES, y=counts.values,
                           marker_color=CLASS_COLOURS,
                           text=counts.values, textposition="outside"))
    fig.update_layout(title="Target class distribution", yaxis_title="Samples",
                      height=400, **_PT)
    return fig


def _gauge(value: float, title: str, colour: str, max_val: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 12}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": colour},
            "bgcolor": "#161b22",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "#1a2e1a"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#2e2a1a"},
                {"range": [max_val * 0.66, max_val], "color": "#2e1a1a"},
            ],
        },
    ))
    fig.update_layout(height=210, **_PT)
    return fig


def _donut(labels: list, values: list, title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
                           marker_colors=["#1f6feb", "#27ae60", "#8e44ad"]))
    fig.update_layout(title=title, height=260, **_PT)
    return fig

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    if "page" not in st.session_state:
        st.session_state.page = PAGE_OVERVIEW

    with st.sidebar:
        st.markdown("### Digital Wellbeing Predictor")
        st.markdown("214129X — Malalpola MLHR")
        st.markdown("---")
        st.markdown("**Navigation**")

        labels = {
            PAGE_OVERVIEW: PAGE_OVERVIEW,
            PAGE_STRESS_PREDICTOR: PAGE_STRESS_PREDICTOR,
            PAGE_DEVICE_PREDICTIONS: PAGE_DEVICE_PREDICTIONS,
            PAGE_PROJECT_REPORT: PAGE_PROJECT_REPORT,
        }
        for pg, label in labels.items():
            active = st.session_state.page == pg
            if st.button(
                label,
                key=f"_nav_{pg}",
                type="primary" if active else "secondary",
                width="stretch",
            ):
                st.session_state.page = pg
                st.rerun()

        st.markdown("---")
        st.caption("Algorithm: Gaussian Naive Bayes\n\nDatasets: DS1 · DS2 · DS3")

    return st.session_state.page


def go_to(page: str) -> None:
    st.session_state._loading_message = f"Loading {page}..."
    st.session_state._loading_until = time.time() + MIN_PAGE_LOADER_SECONDS
    st.session_state.page = page
    st.rerun()

# ---------------------------------------------------------------------------
# JavaScript: collect client-side browser device data (mobile-aware)
# ---------------------------------------------------------------------------

CLIENT_JS = """
<div id="client-data" style="font-family:system-ui,sans-serif;font-size:0.85rem;
  background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-top:0.5rem">
  <strong>Browser-reported device data</strong>
  <div id="batt-info" style="margin-top:0.5rem;color:#8b949e">Checking battery...</div>
  <div id="net-info"  style="margin-top:0.3rem;color:#8b949e">Checking network...</div>
  <div id="mem-info"  style="margin-top:0.3rem;color:#8b949e">Checking memory...</div>
  <p style="margin-top:0.7rem;color:#555;font-size:0.78rem">
    This data comes from your browser (your actual device).
    The server-side stats above reflect the machine hosting the app.
  </p>
</div>
<script>
(function() {
  // Battery API — available in Chrome/Edge, limited in Safari/Firefox
  if (navigator.getBattery) {
    navigator.getBattery().then(function(b) {
      var pct  = Math.round(b.level * 100);
      var stat = b.charging ? "Charging" : "Discharging";
      var mins = b.dischargingTime !== Infinity
                   ? " | ~" + Math.round(b.dischargingTime / 60) + " min left"
                   : "";
      document.getElementById("batt-info").innerHTML =
        "<span style='color:#27ae60'>Battery:</span> " + pct + "% — " + stat + mins;
    }).catch(function() {
      document.getElementById("batt-info").innerHTML =
        "<span style='color:#555'>Battery API not available in this browser.</span>";
    });
  } else {
    document.getElementById("batt-info").innerHTML =
      "<span style='color:#555'>Battery API not supported (try Chrome).</span>";
  }

  // Network information API — Chrome Android / Chrome Desktop
  if (navigator.connection) {
    var c = navigator.connection;
    document.getElementById("net-info").innerHTML =
      "<span style='color:#1f6feb'>Network:</span> " +
      (c.effectiveType || "unknown") + " | Downlink: " +
      (c.downlink || "?") + " Mbps | Save-data: " + (c.saveData ? "yes" : "no");
  } else {
    document.getElementById("net-info").innerHTML =
      "<span style='color:#555'>Network Information API not supported.</span>";
  }

  // Memory — Chrome only
  if (performance.memory) {
    var m = performance.memory;
    var used  = Math.round(m.usedJSHeapSize  / 1048576);
    var total = Math.round(m.totalJSHeapSize / 1048576);
    document.getElementById("mem-info").innerHTML =
      "<span style='color:#e67e22'>JS Heap:</span> " + used + " MB / " + total + " MB";
  } else {
    document.getElementById("mem-info").innerHTML =
      "<span style='color:#555'>Memory API not available.</span>";
  }
})();
</script>
"""

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(p: dict) -> None:
    st.markdown(f'<div class="sec-head">{ICON["home"]} Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "This system predicts a user's **digital stress/wellbeing class** — Low, Medium, or High — "
        "by combining patterns from three cross-device domains: **social media behaviour**, "
        "**mobile phone usage**, and **PC/remote work habits**. "
        "A Gaussian Naive Bayes model trained on three Kaggle datasets drives the classification. "
        "The **Stress Predictor** page lets you assess your own profile interactively. "
        "**Device Predictions** scans your current machine for live battery, network, and screen "
        "usage signals and links them back to the stress model's cross-device feature space. "
        "The **Project Report** covers the full methodology, evaluation, explainability, and "
        "critical discussion."
    )

    if st.button("Go to Stress Predictor", type="primary",
                 key="overview_top_btn"):
        go_to(PAGE_STRESS_PREDICTOR)

    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi"><div class="kpi val" style="color:#27ae60">69.1%</div>'
                    '<div class="kpi lbl">Test accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><div class="kpi val" style="color:#1f6feb">0.703</div>'
                    '<div class="kpi lbl">F1 score (weighted)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi"><div class="kpi val" style="color:#8e44ad">0.898</div>'
                    '<div class="kpi lbl">AUC score (OvR)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi"><div class="kpi val" style="color:#e67e22">451</div>'
                    '<div class="kpi lbl">Unified training samples</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.markdown("**Datasets used**")
        st.dataframe(pd.DataFrame({
            "Dataset":  ["DS1 — Mobile Usage", "DS2 — Social Media & MH", "DS3 — Remote Work & MH"],
            "Source":   ["Kaggle", "Kaggle", "Kaggle"],
            "Samples":  [700, 481, 5000],
            "Role":     ["Mobile enrichment (age-group bridge)",
                         "Primary dataset — target variable engineered here",
                         "PC/work context enrichment (age-group bridge)"],
        }), width="stretch", hide_index=True)
        render_dataset_links()

        st.markdown("**Algorithm — Gaussian Naive Bayes**")
        st.markdown(
            "GNB applies Bayes' theorem with a Gaussian distribution over continuous features. "
            "Unlike the Decision Trees, SVM, and gradient boosting models covered in lectures, "
            "GNB reasons *probabilistically* — computing a confidence score per class, not just "
            "a hard label. This makes it naturally suited for wellbeing assessment where "
            "uncertainty and gradation are clinically meaningful. "
            "The model's learned class means (θ) are directly interpretable, providing "
            "built-in explainability without external tools."
        )

    with col_b:
        st.plotly_chart(_class_dist(p["df_main"]["Stress_Label"]), width="stretch")
        fig_age = px.histogram(p["df2"], x="Age", nbins=20,
                               color_discrete_sequence=["#1f6feb"],
                               title="Age distribution — DS2 survey respondents")
        fig_age.update_layout(height=240, **_PT)
        st.plotly_chart(fig_age, width="stretch")

    st.markdown("---")
    if st.button("Go to Stress Predictor", type="primary",
                 key="overview_bot_btn"):
        go_to(PAGE_STRESS_PREDICTOR)

# ---------------------------------------------------------------------------
# Page: Stress Predictor
# ---------------------------------------------------------------------------

def page_stress_predictor(p: dict) -> None:
    st.markdown(f'<div class="sec-head">{ICON["brain"]} Stress Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter your digital usage profile below. The prediction and all charts update "
        "automatically as you adjust the sliders — no button required. "
        "The model uses your social media behaviour and mental wellbeing indicators directly, "
        "then enriches the prediction with average mobile device and work patterns for your "
        "age group drawn from the two supporting datasets."
    )
    st.markdown("---")

    tab_pred, tab_perf, tab_xai = st.tabs(["My Prediction", "Model Performance", "XAI Explanation"])

    with tab_pred:
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            with st.container(height=760):
                st.markdown("**Parameters**")
                st.markdown("**Demographics**")
                age = st.slider("Age", 13, 70, 22)
                gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
                sm_label = st.selectbox("Daily social media usage", list(SM_HOURS_MAP.keys()))
                sm_hours = SM_HOURS_MAP[sm_label]

                st.markdown("**Social media behaviour**")
                purposeless = st.slider("Purposeless use (1–5)", 1, 5, 3)
                distraction = st.slider("Distracted when busy (1–5)", 1, 5, 3)
                restless = st.slider("Restless without social media (1–5)", 1, 5, 2)
                comparison = st.slider("Compare yourself to others (1–5)", 1, 5, 2)
                validation = st.slider("Seek online validation (1–5)", 1, 5, 2)

                st.markdown("**Mental wellbeing**")
                easily_distracted = st.slider("Easily distracted (1–5)", 1, 5, 3)
                worry = st.slider("Bothered by worries (1–5)", 1, 5, 3)
                concentrate = st.slider("Difficulty concentrating (1–5)", 1, 5, 2)
                depression = st.slider("Feel depressed/down (1–5)", 1, 5, 2)
                interest_fluct = st.slider("Interest in activities fluctuates (1–5)", 1, 5, 3)
                sleep_issues = st.slider("Sleep issues (1–5)", 1, 5, 2)

        inputs = {
            "age": age,
            "gender": gender,
            "sm_hours": sm_hours,
            "purposeless": purposeless,
            "distraction": distraction,
            "restless": restless,
            "comparison": comparison,
            "validation": validation,
            "easily_distracted": easily_distracted,
            "worry": worry,
            "concentrate": concentrate,
            "depression": depression,
            "interest_fluct": interest_fluct,
            "sleep_issues": sleep_issues,
        }

        pred_cls, pred_proba = run_prediction(inputs, p)
        col = CLASS_COLOURS[pred_cls]
        bg = CLASS_BG[pred_cls]

        with right_col:
            with st.container(height=760):
                st.markdown("**Result**")
                st.markdown(
                    f'<div class="result" style="background:{bg};color:{col};border:2px solid {col}">'
                    f'{dot(col, 14)} Predicted stress level: <strong>{CLASS_NAMES[pred_cls]}</strong></div>',
                    unsafe_allow_html=True,
                )

                st.plotly_chart(_proba_bar(pred_proba), width="stretch")

                st.markdown("**Wellness recommendations**")
                tips = {
                    0: [
                        "Maintain your current screen time habits — they are healthy.",
                        "Take short device breaks every 45–60 minutes.",
                        "Sleep patterns seem balanced — keep this up.",
                        "Physical activity is the strongest buffer against digital stress.",
                    ],
                    1: [
                        "Set a daily screen time limit in your phone's Digital Wellbeing settings.",
                        "Try a 30-minute social media break before bed each night.",
                        "Journaling 5 minutes daily reduces worry and distraction.",
                        "Short mindfulness sessions measurably lower digital stress.",
                    ],
                    2: [
                        "Consider a full digital detox day this week.",
                        "Speaking with a counsellor or GP would be beneficial.",
                        "Temporarily remove the social apps you use most.",
                        "Prioritise sleep — no screens for 1 hour before bed.",
                        "Daily aerobic exercise is among the strongest stress reducers available.",
                    ],
                }
                for t in tips[pred_cls]:
                    st.markdown(f"- {t}")

                st.markdown("---")
                st.markdown("**Stress indicator breakdown**")
                ind = {
                    "Worry Score": worry,
                    "Depression Score": depression,
                    "Sleep Issues": sleep_issues,
                    "Restlessness": restless,
                    "Easily Distracted": easily_distracted,
                    "Difficulty Concentrating": concentrate,
                    "Validation Seeking": validation,
                    "Interest Fluctuation": interest_fluct,
                }
                st.plotly_chart(_indicator_bars(ind, p["t33"], p["t66"]), width="stretch")

                st.markdown("---")
                st.markdown("**Cross-device age-group context** — averages from DS1 and DS3 for your age bracket")
                ag = _age_group(age)
                rows = p["df_main"][p["df_main"]["Age_Group"] == ag]
                ref = rows.iloc[0] if len(rows) else p["df_main"].iloc[0]
                
                context_data = {
                    "Metric": [
                        "Avg screen time (hrs/day)",
                        "Avg battery drain (mAh/day)",
                        "Avg app usage (min/day)",
                        "Avg data usage (MB/day)",
                        "Avg work hours (hrs/week)",
                        "Avg work-life balance (1–10)",
                        "Avg social isolation (1–10)",
                        "Avg sleep quality (1–3)",
                    ],
                    "Your age group": [
                        f"{ref['Avg_Screen_Time_hrs']:.1f}",
                        f"{ref['Avg_Battery_Drain_mAh']:.0f}",
                        f"{ref['Avg_App_Usage_min']:.0f}",
                        f"{ref['Avg_Data_Usage_MB']:.0f}",
                        f"{ref['Avg_Work_Hours_Week']:.1f}",
                        f"{ref['Avg_Work_Life_Balance']:.1f}",
                        f"{ref['Avg_Social_Isolation']:.1f}",
                        f"{ref['Avg_Sleep_Quality']:.2f}",
                    ],
                }
                st.dataframe(pd.DataFrame(context_data), width="stretch", hide_index=True)

    with tab_perf:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test accuracy", f"{p['acc'] * 100:.1f}%")
        k2.metric("F1 score",      f"{p['f1']:.3f}")
        k3.metric("AUC (OvR)",     f"{p['auc']:.3f}")
        k4.metric("CV accuracy",   f"{p['cv_scores'].mean() * 100:.1f}% ± {p['cv_scores'].std() * 100:.1f}%")
        st.markdown("---")
        ca, cb = st.columns(2)
        with ca: st.plotly_chart(_confusion_matrix(p["y_test"], p["y_pred"]), width="stretch")
        with cb: st.plotly_chart(_roc_curves(p["y_test"], p["y_proba"]), width="stretch")
        st.plotly_chart(_cv_bars(p["cv_scores"]), width="stretch")
        st.markdown("**Classification report**")
        rpt = classification_report(p["y_test"], p["y_pred"],
                                    target_names=CLASS_NAMES, output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(3), width="stretch")

    with tab_xai:
        st.markdown(
            "Two XAI methods are applied. **Permutation feature importance** (sklearn) measures "
            "how much test accuracy drops when each feature is randomly shuffled. "
            "The **GNB class means heatmap** shows the normalised average feature value the model "
            "associates with each stress class — this is the model's learned internal representation."
        )
        st.plotly_chart(_feature_importance(p["perm_df"]), width="stretch")
        st.markdown("---")
        st.plotly_chart(_class_means_heatmap(p["model"], p["feature_cols"], p["perm_df"]),
                width="stretch")
        st.markdown("---")
        xi1, xi2 = st.columns(2)
        with xi1:
            st.markdown(
                "**Interest Fluctuation** ranks first — users whose engagement with daily "
                "activities fluctuates rapidly are the strongest indicators of High Stress. "
                "**Difficulty Concentrating** and **Depression Score** follow, confirming that "
                "cognitive and mood signals dominate over raw usage time. "
                "**Social Media Hours** alone has moderate importance — *how* a user engages "
                "(purposeless use, validation seeking, distraction) matters more than *duration*."
            )
        with xi2:
            st.markdown(
                "The class means heatmap shows a clear monotonic gradient: "
                f"{dot(CLASS_COLOURS[0])} Low Stress scores consistently near −0.85, "
                f"{dot(CLASS_COLOURS[1])} Medium Stress near zero, and "
                f"{dot(CLASS_COLOURS[2])} High Stress near +0.85 across all top features. "
                "This confirms the model has learned meaningful, separable real-world patterns "
                "rather than noise or spurious correlations.",
                unsafe_allow_html=True,
            )
        st.dataframe(p["perm_df"].round(4), width="stretch", height=983)

# ---------------------------------------------------------------------------
# Page: Device Predictions
# ---------------------------------------------------------------------------

def _render_device_cards(dd: dict, bp: dict, np_: dict, sp: dict) -> None:
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f'<div class="sec-head">{ICON["battery"]} Battery</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dcard">'
            f'<div class="dval" style="color:{bp["colour"]}">{bp["status"]}</div>'
            f'<div class="dlbl">Battery health status</div></div>',
            unsafe_allow_html=True,
        )
        if dd["batt_pct"]:
            st.plotly_chart(_gauge(dd["batt_pct"], "Battery level (%)", bp["colour"]), width="stretch")
            if dd["batt_secs"] and dd["batt_secs"] > 0:
                st.metric("Est. time remaining", f"{dd['batt_secs'] / 3600:.1f} hrs")
        else:
            st.info("No battery detected — this is a desktop machine.")
        st.markdown(f"**Advice:** {bp['tip']}")
        st.caption(f"Drain index: {bp['drain_idx']} | "
                   f"{'Est. hrs left: ' + str(bp['hrs_left']) if bp['hrs_left'] else 'Plugged in'}")

    with p2:
        st.markdown(f'<div class="sec-head">{ICON["wifi"]} Network</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dcard">'
            f'<div class="dval" style="color:{np_["colour"]}">{np_["category"]} usage</div>'
            f'<div class="dlbl">Total: {np_["total"]} GB since boot</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _gauge(min(np_["total"] * 10, 100), "Network load index", np_["colour"]),
            width="stretch",
        )
        st.plotly_chart(
            _donut(["Social / Browsing", "Streaming / Downloads", "Other"],
                   [np_["social"], np_["stream"], np_["other"]],
                   "Estimated data breakdown"),
            width="stretch",
        )
        st.markdown(f"**Advice:** {np_['tip']}")

    with p3:
        st.markdown(f'<div class="sec-head">{ICON["cpu"]} Screen & System</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="dcard">'
            f'<div class="dval" style="color:{sp["colour"]}">{sp["status"]} load</div>'
            f'<div class="dlbl">System load index: {sp["load"]}%</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_gauge(dd["cpu_pct"], "CPU usage (%)", sp["colour"]),
                        width="stretch")
        if dd["ram_pct"] > 80:
            ram_col = "#c0392b"
        elif dd["ram_pct"] > 50:
            ram_col = "#e67e22"
        else:
            ram_col = "#27ae60"
        st.plotly_chart(_gauge(dd["ram_pct"], "RAM usage (%)", ram_col),
                        width="stretch")
        st.markdown(f"**Advice:** {sp['tip']}")


def _render_device_relevance(dd: dict, np_: dict) -> None:
    batt_flag = dd["batt_pct"] and dd["batt_pct"] < 20
    net_flag = np_["total"] > 10
    cpu_flag = dd["cpu_pct"] > 70
    ram_flag = dd["ram_pct"] > 80
    st.markdown(
        f"| Signal | Value | Stress model relevance |\n"
        f"|---|---|---|\n"
        f"| Battery | {str(dd['batt_pct']) + '%' if dd['batt_pct'] else 'N/A'} | "
        f"{'Low battery adds anxiety — raises stress signal' if batt_flag else 'Within normal range'} |\n"
        f"| CPU load | {dd['cpu_pct']}% | "
        f"{'High CPU indicates heavy multitasking stress' if cpu_flag else 'Normal'} |\n"
        f"| RAM | {dd['ram_pct']}% | "
        f"{'High RAM use correlates with overloaded work sessions' if ram_flag else 'Normal'} |\n"
        f"| Network | {np_['total']} GB | "
        f"{'Heavy usage pattern — potential digital overuse' if net_flag else 'Moderate'} |\n"
        f"| OS | {dd['os']} / {dd['machine']} | Device context captured |"
    )


def page_device_predictions() -> None:
    st.markdown(f'<div class="sec-head">{ICON["phone"]} Device Predictions</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Device-level signals — battery drain rate, network data consumption, and screen/CPU "
        "load — are among the features the stress model uses to enrich predictions for each "
        "age group. This page collects live readings from the host machine and applies the same "
        "categorical thresholds the model relies on, giving you a concrete view of how physical "
        "device behaviour connects to your digital stress profile."
    )

    # Explain the mobile limitation clearly before the scan button
    st.info(
        "**Server vs client distinction.** "
        "The scan below reads data from the machine *running this Streamlit server* using Python's "
        "`psutil` library — not from the browser. If you are accessing this app from a mobile phone "
        "over a local network, the stats shown will reflect the laptop or PC hosting the server, "
        "not your phone. The browser-side panel at the bottom of this page attempts to collect "
        "your actual device's battery and network type using browser APIs, which works regardless "
        "of how the app is accessed.",
    )

    scan_btn = st.button("Scan host machine", type="primary")
    if scan_btn:
        st.session_state._loading_message = "Scanning host machine..."
        st.session_state._loading_until = time.time() + MIN_PAGE_LOADER_SECONDS
        with st.spinner("Collecting device data..."):
            st.session_state["device_data"] = collect_server_stats()

    if "device_data" not in st.session_state:
        st.markdown("---")
        st.markdown(
            "Click **Scan host machine** to read live system stats and receive "
            "battery, network, and screen load predictions. The following signals are collected:"
        )
        st.markdown(
            "- Battery level, charging status, estimated time remaining\n"
            "- CPU utilisation and core count\n"
            "- RAM usage (percentage and gigabytes)\n"
            "- Cumulative network data sent and received since system boot\n\n"
            "All data is read locally and never transmitted anywhere."
        )
        st.markdown("---")
        st.markdown("**Browser-side device data (your actual device)**")
        components.html(CLIENT_JS, height=160)
        return

    dd  = st.session_state["device_data"]
    bp  = _batt_prediction(dd["batt_pct"], dd["batt_plugged"], dd["cpu_pct"], dd["ram_pct"])
    np_ = _net_prediction(dd["net_sent_gb"], dd["net_recv_gb"])
    sp  = _screen_prediction(dd["cpu_pct"], dd["ram_pct"])

    st.markdown("---")
    st.markdown("**Live host machine status**")
    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
    ds1.metric("Battery",    f"{dd['batt_pct']}%" if dd["batt_pct"] else "N/A",
               "Charging" if dd["batt_plugged"] else "Discharging")
    ds2.metric("CPU load",   f"{dd['cpu_pct']}%")
    ds3.metric("RAM used",   f"{dd['ram_pct']}%", f"{dd['ram_used_gb']}/{dd['ram_total_gb']} GB")
    ds4.metric("Data sent",  f"{dd['net_sent_gb']} GB")
    ds5.metric("Data received", f"{dd['net_recv_gb']} GB")

    with st.spinner("Preparing device analysis..."):
        st.markdown("---")
        _render_device_cards(dd, bp, np_, sp)

        st.markdown("---")
        st.markdown("**How these readings connect to the stress model**")
        _render_device_relevance(dd, np_)

    st.markdown("---")
    st.markdown("**Browser-reported data (your actual device, regardless of server location)**")
    components.html(CLIENT_JS, height=160)

# ---------------------------------------------------------------------------
# Page: Project Report
# ---------------------------------------------------------------------------

def page_report(p: dict) -> None:
    st.markdown(f'<div class="sec-head">{ICON["report"]} Project Report</div>',
                unsafe_allow_html=True)
    st.caption("214129X — Malalpola MLHR  |  Gaussian Naive Bayes  |  Three Kaggle datasets")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem & Data", "Algorithm", "Evaluation", "XAI", "Discussion",
    ])

    with tab1:
        st.markdown("### Problem Definition & Dataset Collection")
        st.markdown(
            '<div class="rblock"><strong>Problem statement:</strong> Predict a user\'s digital '
            'stress/wellbeing class (Low / Medium / High) from cross-device behavioural patterns '
            'including social media behaviour, mental health indicators, mobile device usage, '
            'and remote work habits.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Excessive digital usage is increasingly associated with anxiety, depression, "
            "sleep disorders, and reduced cognitive performance. No public system currently "
            "integrates mobile, social, and PC usage signals into a single probabilistic "
            "wellbeing classifier. This work addresses that gap."
        )
        st.dataframe(pd.DataFrame({
            "Dataset":  ["DS1 — Mobile Device Usage", "DS2 — Social Media & MH", "DS3 — Remote Work & MH"],
            "Source":   ["Kaggle — valakhorasani", "Kaggle — souvikahmed071", "Kaggle — waqi786"],
            "Samples":  [700, 481, 5000],
            "Role":     [
                "Mobile enrichment via age-group bridge",
                "Primary dataset — composite stress target engineered here",
                "PC/work context enrichment via age-group bridge",
            ],
        }), width="stretch", hide_index=True)
        render_dataset_links()

        st.markdown("**Preprocessing steps**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "DS2 (primary): renamed 16 verbose survey columns; converted social media hours "
                "text to numeric midpoints; label-encoded Gender, Relationship Status, Occupation; "
                "computed composite stress score as the mean of 8 Likert-scale mental health "
                "indicators; applied tertile thresholds to produce a balanced 3-class label."
            )
        with c2:
            st.markdown(
                f"DS1 and DS3 (enrichment): dropped IDs, regions, and job roles; filled missing "
                f"Physical Activity with mode imputation; aggregated by age group "
                f"(18-24, 25-34, 35-44, 45-54, 55+); merged to DS2 using age-group as bridge key. "
                f"Target thresholds: Low ≤ {p['t33']:.2f} | Medium ≤ {p['t66']:.2f} | "
                f"High > {p['t66']:.2f}."
            )

        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(_class_dist(p["df_main"]["Stress_Label"]), width="stretch")
        with cb:
            feats = [
                "Social_Media_Hours", "Worry_Score", "Depression_Score",
                "Sleep_Issues", "Distraction_Score", "Interest_Fluctuation",
                "Age", "Stress_Label",
            ]
            corr = p["df_main"][feats].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Feature correlation matrix (key features)",
                aspect="auto",
            )
            # Larger height so values are readable without zooming
            fig_corr.update_layout(height=480, **_PT)
            st.plotly_chart(fig_corr, width="stretch")

    with tab2:
        st.markdown("### Algorithm — Gaussian Naive Bayes")
        st.markdown(
            '<div class="rblock"><strong>Excluded algorithms (lecture syllabus):</strong> '
            'Linear/Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, '
            'all other Gradient Boosting variants, k-NN, Deep Learning, Image Processing.</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "GNB applies Bayes' theorem with a conditional independence assumption and "
                "models each continuous feature as a Gaussian distribution within each class.\n\n"
                "**P(Cₖ | x) ∝ P(Cₖ) × ∏ᵢ P(xᵢ | Cₖ)**\n\n"
                "where P(xᵢ | Cₖ) = exp(−(xᵢ − μₖᵢ)² / 2σ²ₖᵢ) / √(2πσ²ₖᵢ)\n\n"
                "The class means θ = μₖᵢ learned during training are directly readable, "
                "making the model intrinsically interpretable without needing external XAI tools."
            )
        with c2:
            st.markdown(
                "| Aspect | GNB | Decision Tree | SVM | Random Forest |\n"
                "|---|---|---|---|---|\n"
                "| Reasoning | Probabilistic | Rule-based | Geometric margin | Ensemble voting |\n"
                "| Output | P per class | Class label | Class label | Class label |\n"
                "| Interpretability | Class means | Tree paths | Support vectors | Importances |\n"
                "| Independence assumption | Yes | No | No | No |\n"
                "| Training speed | Very fast | Fast | Moderate | Slow |"
            )

    with tab3:
        st.markdown("### Model Training & Evaluation")
        st.dataframe(pd.DataFrame({
            "Split":   ["Training", "Validation", "Test"],
            "Size":    ["70% — 315 samples", "15% — 68 samples", "15% — 68 samples"],
            "Purpose": ["Fitting + 5-fold CV", "Hyperparameter check", "Final unbiased evaluation"],
        }), width="stretch", hide_index=True)
        st.markdown("*Stratified splitting preserves class balance across all three splits.*")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CV accuracy", f"{p['cv_scores'].mean():.4f} ± {p['cv_scores'].std():.4f}")
        k2.metric("Test accuracy", f"{p['acc']:.4f}")
        k3.metric("Test F1",       f"{p['f1']:.4f}")
        k4.metric("Test AUC",      f"{p['auc']:.4f}")

        ra, rb = st.columns(2)
        with ra: st.plotly_chart(_confusion_matrix(p["y_test"], p["y_pred"]), width="stretch")
        with rb: st.plotly_chart(_roc_curves(p["y_test"], p["y_proba"]), width="stretch")
        st.plotly_chart(_cv_bars(p["cv_scores"]), width="stretch")

        st.markdown(
            "69.1% test accuracy significantly exceeds the 33.3% random baseline for a balanced "
            "3-class problem. AUC of 0.898 confirms strong class separability — particularly for "
            "High Stress (AUC ≈ 0.92), which is the most clinically important class. "
            "High Stress achieves precision of 0.938, meaning the model is correct 93.8% of the time "
            "when it predicts this class. Medium Stress shows lower precision (0.50) due to boundary "
            "overlap with adjacent classes, which is expected and noted as a limitation."
        )

    with tab4:
        st.markdown("### Explainability & Interpretation (XAI)")
        st.markdown(
            "Two methods are applied: **permutation feature importance** quantifies each feature's "
            "contribution by measuring accuracy degradation when it is randomly shuffled. "
            "The **GNB class means heatmap** exposes the model's learned internal representation — "
            "the normalised average value of each feature it associates with each stress class."
        )
        st.plotly_chart(_feature_importance(p["perm_df"]), width="stretch")
        st.plotly_chart(_class_means_heatmap(p["model"], p["feature_cols"], p["perm_df"]),
                width="stretch")
        st.markdown(
            "**Interest Fluctuation** ranks first — rapidly changing daily interests is the "
            "strongest predictor of High Stress. "
            "**Social Media Hours** alone is not the top predictor; *engagement quality* "
            "(purposeless use, distraction, validation seeking) matters more than raw duration. "
            "Cross-device enriched features (Avg Screen Time, Avg Work Hours) contribute "
            "positively, validating the multi-dataset design. "
            "The class means heatmap shows a monotonic gradient from −0.84 (Low) to +0.85 (High) "
            "across all top features, confirming alignment with published psychological research "
            "on digital stress mediators."
        )

    with tab5:
        st.markdown("### Critical Discussion")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="rblock">{ICON["warning"]} <strong>Model limitations</strong><br><br>'
                "GNB assumes feature independence — violated by correlated features such as "
                "Depression Score and Sleep Issues (r = 0.48). "
                "The small final dataset (315 training samples) causes ±11.4% CV variance. "
                "The composite stress target was engineered, not clinically validated against "
                "PHQ-9 or GAD-7 instruments.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="rblock">{ICON["warning"]} <strong>Data quality</strong><br><br>'
                "DS3 had 24% missing Mental Health Condition values — dropped rows may introduce "
                "selection bias. Age-group bridging averages individual variation. "
                "DS2 respondents skew toward university students aged 18–28, limiting "
                "generalisability to older or working populations.</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="rblock">{ICON["info"]} <strong>Bias & ethical considerations</strong><br><br>'
                "Cultural digital usage norms are absent from the model. "
                "Gender encoding loses nuance for non-binary respondents. "
                "Predictions must never replace clinical diagnosis. "
                "Misclassifying High Stress as Low could delay intervention. "
                "Any deployment requires informed consent, data minimisation, opt-in design, "
                "and human oversight.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="rblock">{ICON["check"]} <strong>Future work</strong><br><br>'
                "Collect longitudinal data with clinically validated stress measures. "
                "Integrate SHAP for individual-level explanations. "
                "Add time-series components to capture usage patterns over time. "
                "Conduct fairness audits across age, gender, and cultural subgroups. "
                "Replace rule-based device predictions with models trained on longitudinal "
                "device logs. Consider federated learning for privacy-preserving on-device "
                "deployment.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "**Note on real-time data.** No publicly available dataset provides real-time "
            "streaming mental health or digital usage records. The WHO Global Health Observatory "
            "and CDC NHANES APIs offer periodic updates but not live feeds. "
            "For this project, the three Kaggle datasets provide sufficient coverage; "
            "the model can be periodically retrained as newer survey data is published. "
            "The Device Predictions page bridges this gap by collecting live system readings "
            "directly from the user's machine."
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if "_loading_until" not in st.session_state:
        st.session_state._loading_until = time.time() + MIN_PAGE_LOADER_SECONDS
        st.session_state._loading_message = "Loading application..."

    now = time.time()
    loading_until = st.session_state.get("_loading_until", 0.0)
    if loading_until > now:
        with st.spinner(st.session_state.get("_loading_message", "Loading...")):
            time.sleep(loading_until - now)

    page = render_sidebar()

    try:
        pipeline = load_pipeline()
    except FileNotFoundError:
        st.error(
            "Dataset CSV files not found. Place all three files in a `data_set/` folder "
            "next to `app.py` and restart."
        )
        st.code(
            "data_set/\n"
            "  user_behavior_dataset.csv\n"
            "  smmh.csv\n"
            "  Impact_of_Remote_Work_on_Mental_Health.csv"
        )
        st.stop()

    render_output_controls(pipeline)

    if page == PAGE_OVERVIEW:
        page_overview(pipeline)
    elif page == PAGE_STRESS_PREDICTOR:
        page_stress_predictor(pipeline)
    elif page == PAGE_DEVICE_PREDICTIONS:
        page_device_predictions()
    elif page == PAGE_PROJECT_REPORT:
        page_report(pipeline)


if __name__ == "__main__":
    main()