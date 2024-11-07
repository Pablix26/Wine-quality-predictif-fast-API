# app/main.py

from fastapi import FastAPI
from app.api.endpoints import ml_models_router

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API pour prédire la qualité du vin et interagir avec le modèle prédictif.",
    version="1.0.0"
)

# Inclure le routeur pour les endpoints du modèle
app.include_router(ml_models_router)
