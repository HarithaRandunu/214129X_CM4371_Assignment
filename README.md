# Digital Wellbeing Stress Predictor

MSc AI — Machine Learning and Pattern Recognition Assignment  
Student: **214129X (Malalpola MLHR)**

This project predicts a user's **digital stress level** (`Low`, `Medium`, `High`) by combining three domains:
- social media behavior,
- mobile device usage patterns,
- remote work and mental health context.

It includes:
- a full ML pipeline script for preprocessing, training, evaluation, and explainability,
- a Streamlit front-end for interactive predictions and model interpretation.

---

## 1) Problem Definition & Dataset Collection

### Problem
Excessive and unstructured digital usage can negatively affect mental wellbeing (worry, distraction, sleep issues, mood).  
This system models that risk as a 3-class stress prediction task.

### Datasets (public Kaggle sources)
- **DS1 — Mobile Device Usage and User Behavior**  
  https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset?resource=download
- **DS2 — Social Media and Mental Health**  
  https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health
- **DS3 — Remote Work and Mental Health**  
  https://www.kaggle.com/datasets/waqi786/remote-work-and-mental-health

### Data and preprocessing summary
- Rename long survey column names (DS2) into ML-friendly fields.
- Map social media usage categories to numeric hour midpoints.
- Encode categorical variables (`Gender`, `Relationship_Status`, `Occupation`).
- Build **Composite_Stress** from mental wellbeing indicators.
- Convert stress target into 3 classes via tertiles (`Low/Medium/High`).
- Build enrichment features from DS1 and DS3 grouped by age bands.
- Merge datasets through `Age_Group` bridge and remove incomplete merged rows.
- Standardize input features before modeling.

---

## 2) New Algorithm Selection

The project uses **Gaussian Naive Bayes (GNB)** as the main model.

Why this choice:
- fast and stable for small-to-medium tabular datasets,
- probabilistic outputs per class (useful for risk interpretation),
- interpretable internal parameters (class-wise feature means),
- different from common lecture baselines such as decision trees / logistic regression / k-NN.

---

## 3) Model Training & Evaluation

### Data split
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified splitting to preserve class balance.

### Validation strategy
- 5-fold stratified cross-validation on train split.

### Metrics used
- Accuracy
- Weighted F1 score
- One-vs-Rest AUC
- Classification report
- Confusion matrix
- ROC curves

### Pipeline outputs
Generated into `outputs/`:
- `model_results.csv`
- `feature_importance.csv`

---

## 4) Explainability & Interpretation (XAI)

Implemented methods:
- **Permutation Feature Importance**
- **Partial Dependence Plots (PDP)**
- **GNB Class Means Heatmap**

What this provides:
- identifies most influential features,
- shows directional effect of top numeric features,
- verifies whether learned behavior aligns with domain expectations.

---

## 5) Critical Discussion

Key considerations covered in code/report workflow:
- model assumptions (feature independence in GNB),
- possible quality limitations from source survey datasets,
- potential bias/fairness concerns across demographics,
- ethical use: this is an educational decision-support model, **not** a clinical diagnosis tool.

---

## 6) Bonus: Front-End Integration

A Streamlit application is included in `app.py` with:
- page-based navigation,
- user input panel for stress prediction,
- confidence visualization and tips,
- model performance dashboards,
- XAI visualizations,
- device-side telemetry panel using the **current browser device** (with optional host fallback).

---

## Project Structure

- `app.py` — Streamlit front-end (prediction + XAI + UI pages)
- `ml_pipeline_complete.py` — full training/evaluation/XAI pipeline
- `project_config.py` — shared reusable configuration variables
- `scripts/` — all launcher shortcuts (`.py` and `.cmd`)
- `data_set/` — CSV datasets
- `outputs/` — result CSVs

---

## How to Run

### 1) Install dependencies
```bash
pip install streamlit plotly scikit-learn pandas numpy matplotlib seaborn psutil qrcode[pil]
```

### 2) Run ML pipeline
```bash
python ml_pipeline_complete.py
```

### 3) Run web app
```bash
streamlit run app.py
```

### 4) Python shortcut commands (recommended)
```bash
py -m scripts.run_pipeline
py -m scripts.run_app
py -m scripts.run_checks
```

What they do:
- `py -m scripts.run_pipeline` → runs `ml_pipeline_complete.py`
- `py -m scripts.run_app` → launches Streamlit and prints a terminal QR code for phone access
- `py -m scripts.run_checks` → runs syntax compile checks, then runs pipeline

### 5) Windows `.cmd` shortcuts (shell-specific)

PowerShell:
```bash
.\scripts\run_pipeline.cmd
.\scripts\run_app.cmd
.\scripts\run_checks.cmd
```

Command Prompt (`cmd.exe`):
```bash
scripts\run_pipeline.cmd
scripts\run_app.cmd
scripts\run_checks.cmd
```
