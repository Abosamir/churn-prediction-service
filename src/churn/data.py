import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    """Load the raw data"""

    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop un wanted columns that will not be used by the model
    """

    df = df.copy()
    df = df.drop(columns=['customerID', 'gender', 'PhoneService', "MultipleLines", "TotalCharges"])
    return df

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "raw" / "telcom-customer-churn.csv"

    df = load_data(csv_path)
    print(f"Loaded data: {df.shape}")

    df_clean = clean_data(df)
    print(f"Cleaned data: {df_clean.shape}")
    print(f"Data Columns: {df_clean.columns.to_list()}")