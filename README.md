# Churn Prediction Service

A machine learning pipeline to predict telecommunications customer churn using the IBM Telco dataset. The goal is to identify customers likely to cancel their service so the business can act proactively on retention.

---

## Project Status

| Phase | Status |
|---|---|
| Data Collection | Done |
| Exploratory Data Analysis | Done |
| Feature Engineering | Done |
| Modeling (baseline + model selection) | Done |
| Evaluation & Tuning | In Progress |
| Deployment | Pending |

---

## Dataset

**Source:** IBM Telco Customer Churn (7,043 customers, 21 features)

**Target:** `Churn` — binary label (Yes / No)

**Class distribution:** 73.5% No churn · 26.5% Churn (imbalanced)

| Feature Group | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | CustomerID, tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## Key EDA Findings

**Strong churn predictors (Cramér's V > 0.3):**
- **Contract type** (V=0.41) — month-to-month customers churn at 42.7% vs 1.1% (one-year) and 2.8% (two-year)
- **OnlineSecurity / TechSupport** (V≈0.34) — customers without these services churn significantly more
- **InternetService** (V=0.32) — fiber optic users churn more than DSL users
- **PaymentMethod** (V=0.30) — electronic check users churn at higher rates

**Weak predictors (likely to drop):**
- `gender` (V=0.008) — ~27% churn regardless of gender
- `PhoneService` (V=0.011) — minimal discriminative power

**Data quality fixes applied:**
- 11 blank entries in `TotalCharges` (whitespace strings) → imputed with `MonthlyCharges` for new customers with 0 tenure
- `TotalCharges` cast from `object` to `float64`

---

## Baseline Model

A `DummyClassifier` (most-frequent) and a Logistic Regression pipeline were trained on `data/processed/cleaned.csv` as a baseline to beat:

<div align="center">

| Metric | Dummy | LogReg | Delta |
|:-------|:-----:|:------:|:-----:|
| Accuracy  | 0.7346 | 0.7977 | +0.0631 |
| Precision | 0.0000 | 0.6422 | — |
| Recall    | 0.0000 | 0.5374 | — |
| F1 Score  | 0.0000 | 0.5852 | — |
| ROC AUC   | 0.5000 | 0.8383 | +0.3383 |

</div>

Recall is the key metric to optimize: a missed churner (false negative) costs the business the customer's full lifetime value (~$300–$1000).

---

## Feature Engineering

Five engineered features were added on top of the cleaned data:

| Feature | Rationale |
|---|---|
| `tenure_bucket` | tenure is the strongest predictor; relationship is likely non-linear |
| `services_count` | captures customers subscribed to many vs. few services |
| `is_new_customer` | flags tenure < 6 months |
| `has_auto_payment` | automatic-payment customers tend to churn less |
| `monthly_charge_per_tenure` | normalizes spend against customer lifetime |

With 5-fold stratified cross-validation, these features and three model families were compared on ROC AUC:

<div align="center">

| Model | mean AUC | std | n_folds |
|:-------|:-----:|:------:|:-----:|
| LogReg (no features) | 0.8442 | 0.0114 | 5 |
| LogReg (with features) | **0.8476** | 0.0114 | 5 |
| RandomForest (with features) | 0.8458 | 0.0093 | 5 |
| GradientBoosting (with features) | 0.8495 | 0.0120 | 5 |

</div>

**Decision:** Logistic Regression (with engineered features) was selected to carry forward into tuning. The four models score within noise of each other on AUC, so the tie-break favored interpretability, deployability, and training speed — all of which favor LogReg over the ensemble models.

---

## Project Structure

```
churn-prediction-service/
├── data/
│   ├── raw/
│   │   └── telcom-customer-churn.csv
│   └── processed/
│       └── cleaned.csv
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline.ipynb       # Dummy vs. logistic regression baseline
│   ├── 03_feature_eng.ipynb    # Engineered features, single-split comparison
│   └── 04_model_tuning.ipynb   # Cross-validated model selection
├── src/                         # Source modules (in progress)
├── tests/                       # Test suite (in progress)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Stack

| Category | Libraries |
|---|---|
| ML | scikit-learn 1.5.2, XGBoost 2.1.1, LightGBM 4.5.0 |
| Data | pandas 2.2.3, numpy 1.26.4, scipy 1.13.1 |
| Visualization | matplotlib 3.9.2, seaborn 0.13.2, plotly 5.24.1 |
| Notebooks | Jupyter, JupyterLab |

**Python:** 3.11+

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd churn-prediction-service

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Open the notebooks in order to follow the pipeline from EDA through model selection:

```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## Next Steps

1. **Hyperparameter tuning** — tune the selected LogReg (+ engineered features) pipeline, e.g. via `GridSearchCV`
2. **Class imbalance** — try `class_weight='balanced'` or SMOTE and compare against the current untreated baseline
3. **Threshold tuning** — pick a decision threshold that favors recall, given the cost asymmetry of missed churners
4. **Evaluation** — confirm gains hold via precision-recall curves, not just ROC AUC
5. **Deployment** — package the winning pipeline under `src/` with tests, and expose it as a service
