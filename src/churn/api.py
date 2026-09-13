# imports (FastAPI, Pydantic, your churn modules)
import pandas as pd 
from pathlib import Path
from churn.predict import load_model, predict
from churn.features import add_features

from fastapi import FastAPI
from pydantic import BaseModel
# load model once at startup
repo_root = Path(__file__).resolve().parents[2]

models_root = repo_root / "models"
models_file_path = models_root / "model.pkl"

pipe = load_model(models_file_path) 
# Pydantic model for input validation (one customer's data)
class InputData(BaseModel):
    SeniorCitizen: int 
    Partner: str 
    Dependents: str 
    tenure: int 
    InternetService: str 
    OnlineSecurity: str 
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingTV: str 
    StreamingMovies: str 
    Contract: str 
    PaperlessBilling: str 
    PaymentMethod: str 
    MonthlyCharges: float


app = FastAPI()

@app.post('/predict')
def predict_endpoint(customer: InputData):
    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df)
    probability = predict(pipe, df)

    return {"churn_probability": float(probability[0])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)