# imports (FastAPI, Pydantic, your churn modules)
import FastA

# load model once at startup

# Pydantic model for input validation (one customer's data)
#   — all 15 feature columns that clean_data outputs (before add_features)
#   — think about types: tenure is int, MonthlyCharges is float,
#     Contract is str, SeniorCitizen is int (0/1), etc.

# POST endpoint: /predict
#   — receive customer JSON
#   — convert to DataFrame (one row)
#   — add_features
#   — predict
#   — return {"churn_probability": float}