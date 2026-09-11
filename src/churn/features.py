import pandas as pd
from pathlib import Path

from churn.data import load_data, clean_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """add binary, nominal and numerical columns introduced in 03/05"""
    
    df = df.copy()
    # 1. Tenure column
    bin_edges = [0, 12, 24, 48, 72]
    bin_labels = ['0-12', '13-24', '25-48', '49-72']

    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # 2. services column
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    # axis = 1, to sum them horizontally
    df["services_count"] = (df[services]=="Yes").sum(axis=1)

    # 3. check new customers
    def check_new_customer(tenure):
        if tenure < 6:
            return 1
        else:
            return 0
    df["is_new_customer"] = df["tenure"].map(check_new_customer)

    # 4. automatic payment or not 
    def check_automatic_payment(payment_method):
        if "automatic" in payment_method:
            return 1
        else:
            return 0
    df["has_auto_payment"] = df["PaymentMethod"].map(check_automatic_payment)

    # 5. monthly_charge_per_tenure relation from both columns MonthlyCharges and tenure
    df["monthly_charge_per_tenure"] = (df["MonthlyCharges"]/(df["tenure"]+1))

    # binary map before implementing column transformer
    binary_cols = ["Partner", "Dependents", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df

def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scaler on numerics, OHE on nominals, binary map already applied."""


    num_cols = ["tenure", "MonthlyCharges","services_count", "monthly_charge_per_tenure"]
    nominal_cols = ["InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaymentMethod", "tenure_bucket"]


    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), nominal_cols)
    ], remainder="passthrough")

    return preprocessor



if __name__ == "__main__":

    repo_path = Path(__file__).resolve().parents[2]
    csv_path = repo_path / "data" / "raw" / "telcom-customer-churn.csv"

    df = clean_data(load_data(csv_path))

    df = add_features(df)
    X = df.drop(columns=['Churn'])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    print(f"Shape: {transformed.shape}")