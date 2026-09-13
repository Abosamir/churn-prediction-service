# imports (FastAPI, Pydantic, your churn modules)
import pandas as pd 
from pathlib import Path
from churn.predict import load_model, predict
from churn.features import add_features

from fastapi import FastAPI
from typing import Literal, get_args, get_type_hints
from pydantic import BaseModel, Field, model_validator
# load model once at startup
repo_root = Path(__file__).resolve().parents[2]

models_root = repo_root / "models"
models_file_path = models_root / "model.pkl"

pipe = load_model(models_file_path) 
# Pydantic model for input validation (one customer's data)
class InputData(BaseModel):
    SeniorCitizen: Literal[0, 1]
    Partner: Literal['Yes','No']
    Dependents: Literal['Yes','No']
    tenure: int = Field(gt=0, le=100)
    InternetService: Literal['DSL','Fiber optic','No']
    OnlineSecurity: Literal['No','Yes','No internet service']
    OnlineBackup: Literal['No','Yes','No internet service']
    DeviceProtection: Literal['No','Yes','No internet service']
    TechSupport: Literal['No','Yes','No internet service']
    StreamingTV: Literal['No','Yes','No internet service']
    StreamingMovies: Literal['No','Yes','No internet service']
    Contract: Literal['Month-to-month','One year','Two year']
    PaperlessBilling: Literal['Yes','No']
    PaymentMethod: Literal['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']
    MonthlyCharges: float = Field(gt=0, le=200)

    @model_validator(mode="before")
    @classmethod
    def normalize_strings(cls, data):
        if not isinstance(data, dict):
            return data

        hints = get_type_hints(cls)
        for field, hint in hints.items():
            allowed = get_args(hint)
            if field not in data or not allowed:
                continue

            value = data[field]
            if not isinstance(value, str):
                continue

            # find case-insensitive match among allowed string values
            for option in allowed:
                if isinstance(option, str) and option.lower() == value.lower():
                    data[field] = option
                    break

        return data


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