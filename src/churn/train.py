# imports
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from churn.data import load_data, clean_data
from churn.features import add_features, build_preprocessor
from sklearn.metrics import roc_auc_score
import joblib

def train(csv_path):
    """this function is made to take the path of the Churn data and making all the processing required until building and saving the trained model

    Args:
        csv_path (pathlib): path to the churn data
    """

    # load, clean and add feature to the data
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_features(df)

    # convert the target label to binary encoding
    df['Churn'] = df['Churn'].map({"No": 0, "Yes": 1})

    # split X and y
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # build pipeline (preprocessor + LogisticRegression)
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # fit on train only
    pipe.fit(X_train, y_train)

    # predict probabilities on test and train
    y_prob_test = pipe.predict_proba(X_test)[:, 1]
    y_prob_train = pipe.predict_proba(X_train)[:, 1]

    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    roc_auc_train = roc_auc_score(y_train, y_prob_train)
        
    print(f"\n{'='*40}")
    print(f" ROC-AUC on Train: {roc_auc_train:.4f}")
    print(f" ROC-AUC on Test: {roc_auc_test:.4f}")
    print(f"{'='*40}")

    # save fitted pipeline to models/model.pkl
    repo_root = Path(__file__).resolve().parents[2]

    models_root = repo_root / "models"
    models_file_path = models_root / "model.pkl"
    
    models_root.mkdir(parents=True, exist_ok=True) 
    
    joblib.dump(pipe, models_file_path)
    print(f"Model saved successfully at: {models_file_path}")

# main guard

if __name__=="__main__":
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "raw" / "telcom-customer-churn.csv"
    train(csv_path)