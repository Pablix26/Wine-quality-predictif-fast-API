# app/test/test_ml_models.py

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status
from app.main import app  # Assurez-vous que 'app' est correctement importé
import os
import shutil
import numpy as np
import pandas as pd
import tempfile

# Chemins vers les fichiers de données et modèles
from app.config import DATA_PATH, MODEL_PATH, MODEL_INFO_PATH

# Fixture pour sauvegarder et restaurer le fichier de données
@pytest.fixture(scope="module", autouse=True)
def backup_data():
    # Créer un répertoire temporaire
    temp_dir = tempfile.mkdtemp()
    temp_data_path = os.path.join(temp_dir, 'Wines.csv')
    # Copier le fichier de données dans le répertoire temporaire
    shutil.copy(DATA_PATH, temp_data_path)
    # Modifier le chemin de données dans le module ml_models
    import app.api.endpoints.ml_models as ml_models
    ml_models.DATA_PATH = temp_data_path
    # Recharger les données
    ml_models.data = pd.read_csv(ml_models.DATA_PATH)
    ml_models.X_data = ml_models.data.drop(['quality', 'Id'], axis=1)
    # Fournir le chemin temporaire aux tests
    yield
    # Supprimer le répertoire temporaire après les tests
    shutil.rmtree(temp_dir)
    # Supprimer le modèle entraîné
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    # Supprimer les informations du modèle
    if os.path.exists(MODEL_INFO_PATH):
        os.remove(MODEL_INFO_PATH)

@pytest.mark.asyncio
async def test_predict_quality():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        payload = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }
        response = await ac.post("/api/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "quality_score" in data
    assert isinstance(data["quality_score"], float)
    assert 0 <= data["quality_score"] <= 10

@pytest.mark.asyncio
async def test_get_perfect_wine():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/api/predict")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "predicted_quality_score" in data
    assert 0 <= data["predicted_quality_score"] <= 10

@pytest.mark.asyncio
async def test_get_model():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/api/model")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/octet-stream"

@pytest.mark.asyncio
async def test_get_model_description():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/api/model/description")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "parameters" in data
    assert "performance" in data

@pytest.mark.asyncio
async def test_add_new_data():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        payload = {
            "fixed_acidity": 7.5,
            "volatile_acidity": 0.6,
            "citric_acid": 0.05,
            "residual_sugar": 2.0,
            "chlorides": 0.07,
            "free_sulfur_dioxide": 15.0,
            "total_sulfur_dioxide": 40.0,
            "density": 0.9970,
            "pH": 3.5,
            "sulphates": 0.6,
            "alcohol": 10.0,
            "quality": 6
        }
        response = await ac.put("/api/model", json=payload)
    assert response.status_code == status.HTTP_200_OK, f"Response text: {response.text}"
    data = response.json()
    assert "message" in data
    assert data["message"] == "Nouvelle donnée ajoutée avec succès."
    assert "Id" in data
    assert isinstance(data["Id"], int)

@pytest.mark.asyncio
async def test_retrain_model():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/api/model/retrain")
    assert response.status_code == status.HTTP_200_OK, f"Response text: {response.text}"
    data = response.json()
    assert "message" in data
    assert data["message"] == "Le modèle a été réentraîné avec succès."
    assert "performance" in data
    assert "mean_squared_error" in data["performance"]
    assert "r2_score" in data["performance"]
    assert "accuracy_within_10_percent (%)" in data["performance"]

@pytest.mark.asyncio
async def test_evaluate_model():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/api/model/evaluation")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "model_name" in data
    assert "parameters" in data
    assert "performance" in data
    assert "mean_squared_error" in data["performance"]
    assert "r2_score" in data["performance"]
    assert "mean_absolute_percentage_error (%)" in data["performance"]
    assert "accuracy_within_10_percent (%)" in data["performance"]
    # Optionnellement, vérifier les types
    assert isinstance(data["model_name"], str)
    assert isinstance(data["parameters"], dict)
    assert isinstance(data["performance"]["mean_squared_error"], float)
    assert isinstance(data["performance"]["r2_score"], float)
    assert isinstance(data["performance"]["mean_absolute_percentage_error (%)"], float)
    assert isinstance(data["performance"]["accuracy_within_10_percent (%)"], float)

