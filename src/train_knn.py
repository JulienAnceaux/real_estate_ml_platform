# src/train_knn.py

from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from db import read_sql

# Dossier pour stocker le modèle
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "knn_offices.joblib"
INDEX_PATH = MODELS_DIR / "knn_offices_index.npy"  # pour garder l'ordre des OfficeID


def load_training_data():
    # On se base sur ta vue de training
    query = "SELECT * FROM report.vTraining_Offices_ML"
    df = read_sql(query)

    print("Colonnes disponibles dans la vue :")
    print(df.columns)

    # 🔴 Adapte cette liste si besoin
    feature_cols = ["Lat", "Lng"]

    # Diagnostic des NaN
    print("\nNombre de NaN par colonne de features :")
    print(df[feature_cols].isna().sum())

    # On supprime les lignes qui ont au moins un NaN dans les features
    df_clean = df.dropna(subset=feature_cols)

    nb_dropped = df.shape[0] - df_clean.shape[0]
    if nb_dropped > 0:
        print(f"\n⚠️ {nb_dropped} lignes supprimées à cause de NaN dans {feature_cols}")

    # On prépare les matrices X et y
    X = df_clean[feature_cols].astype(float).values
    office_ids = df_clean["OfficeID"].astype(int).values

    return X, office_ids


def train_knn():
    X, office_ids = load_training_data()

    model = NearestNeighbors(
        n_neighbors=10,   # configurable plus tard
        metric="euclidean"
    )
    model.fit(X)

    # Sauvegarde du modèle + index
    joblib.dump(model, MODEL_PATH)
    np.save(INDEX_PATH, office_ids)

    print(f"✅ Modèle KNN sauvegardé dans {MODEL_PATH}")
    print(f"✅ Index OfficeID sauvegardé dans {INDEX_PATH}")


if __name__ == "__main__":
    train_knn()
