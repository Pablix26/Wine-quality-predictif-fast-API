# app/api/endpoints/ml_models.py

import os
import json
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Charger les données
from joblib import dump, load

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from scipy.optimize import minimize

# Charger le modèle
from app.config import DATA_PATH, MODEL_PATH, MODEL_INFO_PATH

# Initialiser le routeur
router = APIRouter()

# Configurer les logs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Définir les modèles Pydantic
from app.api.endpoints.model import WineFeatures, NewWineData

# Fonction pour calculer l'Accuracy Within Percentage (AWP)
def calculate_awp(y_true, y_pred, threshold=0.1):
    """
    Calcule le pourcentage de prédictions où l'erreur relative est inférieure ou égale au seuil.
    :param y_true: Valeurs réelles
    :param y_pred: Valeurs prédites
    :param threshold: Seuil de tolérance (par défaut 10%)
    :return: Pourcentage d'accuracy
    """
    # Eviter la division par zéro
    mask = y_true != 0
    relative_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    awp = np.mean(relative_errors <= threshold) * 100
    return awp

# Fonction pour entraîner le modèle avec optimisation
def train_model():
    # Charger le jeu de données
    data = pd.read_csv(DATA_PATH)
    
    # Sous-échantillonner si le dataset est trop grand
    if len(data) > 10000:
        data = data.sample(n=10000, random_state=42)
        logger.info("Jeu de données sous-échantillonné à 10 000 enregistrements.")
    
    # Gérer les valeurs manquantes en les imputant avec la moyenne
    data_filled = data.fillna(data.mean())
    
    # Sauvegarder les données nettoyées
    data_filled.to_csv(DATA_PATH, index=False)
    
    # Vérifier s'il y a encore des valeurs manquantes
    if data_filled.isnull().values.any():
        raise ValueError("Le jeu de données contient encore des valeurs manquantes après l'imputation.")
    
    # Préparer les données
    X = data_filled.drop(['quality', 'Id'], axis=1)
    y = data_filled['quality']
    
    # Diviser les données (si nécessaire, sinon entrainer sur tout le dataset)
    # Ici, nous utilisons la validation croisée, donc pas de division explicite nécessaire
    
    # Définir le modèle de base
    base_model = RandomForestRegressor(random_state=42)
    
    # Définir la grille de paramètres pour RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    
    logger.info("Début de la recherche des hyperparamètres avec RandomizedSearchCV...")
    
    # Définir le RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,        # Nombre de combinaisons aléatoires à essayer
        cv=3,             # Nombre de folds
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Effectuer la recherche des hyperparamètres
    random_search.fit(X, y)
    
    # Meilleurs paramètres et score
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info(f"Meilleurs paramètres : {best_params}")
    logger.info(f"Meilleur score R² lors de la validation croisée : {best_score:.4f}")
    
    # Entraîner le modèle avec les meilleurs paramètres sur l'ensemble des données
    best_model = random_search.best_estimator_
    best_model.fit(X, y)
    
    # Évaluer le modèle sur l'ensemble des données
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100  # Convertir en pourcentage
    awp = calculate_awp(y.values, y_pred, threshold=0.1)    # 10%
    
    # Enregistrer le modèle
    dump(best_model, MODEL_PATH)
    
    # Enregistrer les informations du modèle
    model_info = {
        "model_name": "RandomForestRegressor",
        "parameters": best_model.get_params(),
        "performance": {
            "mean_squared_error": mse,
            "r2_score": r2,
            "mean_absolute_percentage_error (%)": mape,
            "accuracy_within_10_percent (%)": awp
        }
    }
    with open(MODEL_INFO_PATH, 'w') as f:
        json.dump(model_info, f)
    
    logger.info("Modèle RandomForestRegressor entraîné et sauvegardé avec succès.")

# Charger le modèle (entraîner si nécessaire)
if not os.path.exists(MODEL_PATH):
    logger.info("Modèle non trouvé. Entraînement du modèle...")
    train_model()
    logger.info("Modèle entraîné et sauvegardé.")

model = load(MODEL_PATH)

# Charger les données pour les statistiques (utilisé dans GET /api/predict)
data = pd.read_csv(DATA_PATH)
X_data = data.drop(['quality', 'Id'], axis=1)

# Endpoint POST /api/predict pour réaliser une prédiction
@router.post("/api/predict")
async def predict_quality(features: WineFeatures):
    try:
        logger.info(f"Données reçues pour prédiction : {features.model_dump()}")
        
        # Créer un DataFrame avec les données reçues
        input_df = pd.DataFrame([features.model_dump()])
        logger.info(f"DataFrame créé :\n{input_df}")
        
        # Vérifier que les colonnes correspondent
        if not list(X_data.columns) == list(input_df.columns):
            raise ValueError(f"Les colonnes d'entrée ne correspondent pas aux colonnes du modèle. Colonnes attendues : {X_data.columns.tolist()}")
        
        # Sélectionner les colonnes dans le même ordre que lors de l'entraînement
        input_data = input_df[X_data.columns]
        logger.info(f"DataFrame sélectionné pour prédiction :\n{input_data}")
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        score_sur_10 = max(0, min(10, prediction))
        logger.info(f"Prédiction effectuée : {prediction}, Score sur 10 : {score_sur_10}")
        return {"quality_score": round(score_sur_10, 2)}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint GET /api/predict pour générer le "vin parfait"
@router.get("/api/predict")
async def get_perfect_wine():
    try:
        mean_values = X_data.mean()
        std_values = X_data.std()

        def objective_function(features):
            prediction = model.predict([features])[0]  # Extraire le premier élément
            return -prediction  # Retourne un float

        initial_guess = mean_values.values
        bounds = [(mean - 2*std, mean + 2*std) for mean, std in zip(mean_values, std_values)]

        result = minimize(objective_function, initial_guess, bounds=bounds)

        if result.success:
            perfect_wine_features = result.x
            predicted_quality = -result.fun  # Accès corrigé
            score_sur_10 = max(0, min(10, predicted_quality))
            feature_names = X_data.columns
            features_dict = dict(zip(feature_names, perfect_wine_features))
            features_dict["predicted_quality_score"] = round(score_sur_10, 2)
            return features_dict
        else:
            raise HTTPException(status_code=500, detail=f"Impossible de générer le vin parfait : {result.message}")
    except Exception as e:
        logger.error(f"Erreur lors de la génération du vin parfait : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint GET /api/model pour obtenir le modèle sérialisé
@router.get("/api/model")
async def get_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(MODEL_PATH, media_type='application/octet-stream', filename='wine_quality_model.joblib')
    else:
        logger.error("Modèle non trouvé lors de la tentative de téléchargement.")
        raise HTTPException(status_code=404, detail="Modèle non trouvé.")

# Endpoint GET /api/model/description pour obtenir des informations sur le modèle
@router.get("/api/model/description")
async def get_model_description():
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        return model_info
    else:
        logger.error("Informations du modèle non trouvées lors de la demande.")
        raise HTTPException(status_code=404, detail="Informations du modèle non trouvées.")

# Endpoint PUT /api/model pour ajouter une nouvelle donnée au jeu de données
@router.put("/api/model")
async def add_new_data(new_data: NewWineData):
    try:
        # Charger le jeu de données existant
        data = pd.read_csv(DATA_PATH)

        # Générer un nouvel Id si nécessaire
        if new_data.Id is None:
            new_id = int(data['Id'].max()) + 1 if not data.empty else 1
            new_data.Id = new_id
        else:
            new_data.Id = int(new_data.Id)

        # Convertir les types numpy en types natifs
        new_data_dict = new_data.model_dump()
        for key, value in new_data_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                new_data_dict[key] = value.item()

        # Ajouter la nouvelle entrée
        new_row = pd.DataFrame([new_data_dict])
        data = pd.concat([data, new_row], ignore_index=True)

        # Enregistrer le jeu de données mis à jour
        data.to_csv(DATA_PATH, index=False)

        logger.info(f"Nouvelle donnée ajoutée avec l'Id : {new_data.Id}")
        return {"message": "Nouvelle donnée ajoutée avec succès.", "Id": new_data.Id}
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de la nouvelle donnée : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint POST /api/model/retrain pour réentraîner le modèle
@router.post("/api/model/retrain")
async def retrain_model():
    try:
        logger.info("Démarrage du réentraînement du modèle.")
        train_model()
        logger.info("Modèle réentraîné avec succès.")

        # Recharger le modèle entraîné
        global model
        model = load(MODEL_PATH)

        # Charger les nouvelles informations du modèle
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
        else:
            model_info = {}

        return {"message": "Le modèle a été réentraîné avec succès.", "performance": model_info.get("performance", {})}
    except Exception as e:
        logger.error(f"Erreur lors du réentraînement du modèle : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint GET /api/model/evaluation pour évaluer le modèle
# Endpoint GET /api/model/evaluation pour évaluer le modèle
@router.get("/api/model/evaluation")
async def evaluate_model():
    try:
        # Charger les informations du modèle depuis model_info.json
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Informations du modèle non trouvées.")
        
        model_name = model_info.get("model_name", "Unknown")
        parameters = model_info.get("parameters", {})
        
        # Extraire les caractéristiques et les valeurs réelles de qualité
        X = X_data
        y_true = data['quality']
        
        # Faire les prédictions
        y_pred = model.predict(X)
        
        # Calculer le MAPE
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convertir en pourcentage
        
        # Calculer le MSE et le R² Score
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculer l'Accuracy Within 10%
        awp = calculate_awp(y_true.values, y_pred, threshold=0.1)  # 10%
        
        # Préparer les métriques de performance
        performance = {
            "mean_absolute_percentage_error (%)": round(mape, 2),
            "mean_squared_error": round(mse, 4),
            "r2_score": round(r2, 4),
            "accuracy_within_10_percent (%)": round(awp, 2)
        }
        
        # Retourner le nom du modèle, ses paramètres et les performances
        return {
            "model_name": model_name,
            "parameters": parameters,
            "performance": performance
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle : {e}")
        raise HTTPException(status_code=500, detail=str(e))

