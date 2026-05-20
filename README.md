# Churn Prediction Service

A machine learning pipeline to predict telecommunications customer churn using the IBM Telco dataset. The goal is to identify customers likely to cancel their service so the business can act proactively on retention.

---

## Project Status

| Phase | Status |
|---|---|
| Data Collection | Done |
| Exploratory Data Analysis | Done |
| Feature Engineering | Pending |
| Modeling | Pending |
| Evaluation & Tuning | Pending |
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

## Project Structure

```
churn-prediction-service/
├── data/
│   └── raw/
│       └── telcom-customer-churn.csv
├── notebooks/
│   └── 01_eda.ipynb          # Full exploratory data analysis
├── src/                       # Source modules (in progress)
├── tests/                     # Test suite (in progress)
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

Open the EDA notebook to explore the dataset:

```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## Modeling Plan

Based on EDA findings, the next steps are:

1. **Feature engineering** — drop `gender`, `customerID`, `PhoneService`; evaluate dropping `TotalCharges` (collinear with tenure × MonthlyCharges)
2. **Encoding** — one-hot encode nominal categoricals; binary-map yes/no columns
3. **Scaling** — normalize `tenure` and `MonthlyCharges`
4. **Class imbalance** — baseline first, then `class_weight='balanced'` or SMOTE if needed
5. **Model selection** — XGBoost or LightGBM as primary candidates
6. **Evaluation** — AUC-ROC, F1, and precision-recall curves (appropriate for imbalanced data)

---

## Author

Mohamed Samir
