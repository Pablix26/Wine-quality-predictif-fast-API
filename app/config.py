# app/config.py

import os

# Déterminer le répertoire de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le fichier de données
DATA_PATH = os.path.join(BASE_DIR, 'datasource', 'Wines.csv')
print (DATA_PATH)

# Dossier pour les modèles
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Assurer que le dossier des modèles existe
os.makedirs(MODEL_DIR, exist_ok=True)

# Chemins vers les fichiers du modèle
MODEL_PATH = os.path.join(MODEL_DIR, 'wine_quality_model.joblib')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')
