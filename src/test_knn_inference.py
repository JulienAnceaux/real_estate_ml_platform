# src/test_knn_inference.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from db import read_sql

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "knn_offices.joblib"
INDEX_PATH = MODELS_DIR / "knn_offices_index.npy"

FEATURE_COLS = ["Lat", "Lng", "GLA"]


def load_training_data():
    df = read_sql("SELECT * FROM report.vTraining_Offices_ML")
    df_clean = df.dropna(subset=FEATURE_COLS)

    X = df_clean[FEATURE_COLS].astype(float).values
    office_ids = df_clean["OfficeID"].astype(int).values
    return df_clean, X, office_ids


def load_model():
    model: NearestNeighbors = joblib.load(MODEL_PATH)
    office_ids = np.load(INDEX_PATH)
    return model, office_ids


def test_for_office(office_id: int, top_n: int = 5):
    # On recharge les données de training pour avoir les infos complètes (nom, ville, etc.)
    df_clean, X, office_ids = load_training_data()

    # On récupère la ligne correspondant à l'OfficeID demandé
    row = df_clean[df_clean["OfficeID"] == office_id]
    if row.empty:
        raise ValueError(f"Aucun immeuble trouvé pour OfficeID = {office_id}")

    x_query = row[FEATURE_COLS].astype(float).values  # shape (1, 3)

    model, model_office_ids = load_model()

    distances, indices = model.kneighbors(x_query, n_neighbors=top_n)
    distances = distances[0]
    indices = indices[0]

    # On mappe les indices des voisins vers les OfficeID réels
    neighbor_ids = model_office_ids[indices]

    # On récupère les infos des voisins dans df_clean
    neighbors = df_clean[df_clean["OfficeID"].isin(neighbor_ids)].copy()

    # On réordonne selon le rang KNN
    order = {oid: rank for rank, oid in enumerate(neighbor_ids, start=1)}
    neighbors["Rank"] = neighbors["OfficeID"].map(order)
    neighbors["Distance"] = neighbors["OfficeID"].map(
        {oid: dist for oid, dist in zip(neighbor_ids, distances)}
    )

    neighbors = neighbors.sort_values("Rank")

    cols_to_show = ["Rank", "OfficeID", "AssetName", "City", "GLA", "Lat", "Lng", "Distance"]
    print(neighbors[cols_to_show])


if __name__ == "__main__":
    # 🔢 Choisis un OfficeID qui existe dans ta vue
    test_for_office(office_id=1, top_n=5)
