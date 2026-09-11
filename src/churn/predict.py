import joblib
from pathlib import Path

from churn.data import load_data, clean_data
from churn.features import add_features

def load_model(model_path):
    """Load a saved model 

    Args:
        model_path (pathlib): path to a saved trained model

    Returns:
        Pipeline: the trained model
    """

    return joblib.load(model_path)

def predict(pipe, df):
    """This is to return probability given data and trained model

    Args:
        pipe (Pipeline): the trained model
        df (pd.DataFrame): the new customer data to check 

    Returns:
        float: probability that the customer will churn
    """

    return pipe.predict_proba(df)[:, 1]

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "raw" / "telcom-customer-churn.csv"

    models_root = repo_root / "models"
    models_file_path = models_root / "model.pkl"

    pipe = load_model(models_file_path) 

    # load, clean and add feature to the data
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_features(df)


    # split X and y
    X = df.drop(columns=['Churn'])

    probabilities = predict(pipe, X[:5])

    print(f"The probabilities that the first 5 customers will churn is: \n {probabilities}")