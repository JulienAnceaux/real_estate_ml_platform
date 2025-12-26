# src/run_comps.py

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


def load_model():
    model: NearestNeighbors = joblib.load(MODEL_PATH)
    office_ids = np.load(INDEX_PATH)
    return model, office_ids


def load_training_universe() -> pd.DataFrame:
    """
    Universe des comparables (marché).
    On prend la vue report.vTraining_Offices_ML (déjà ML-friendly).
    """
    df = read_sql("SELECT * FROM report.vTraining_Offices_ML")
    # Sécurité : on drop les lignes incomplètes sur les features
    df = df.dropna(subset=FEATURE_COLS)

    # Cast explicite pour éviter les surprises (DECIMAL -> object parfois)
    df["Lat"] = df["Lat"].astype(float)
    df["Lng"] = df["Lng"].astype(float)
    df["GLA"] = df["GLA"].astype(float)
    df["OfficeID"] = df["OfficeID"].astype(int)

    return df


def load_project(project_id: int) -> pd.Series:
    """
    Projet (subject) : on utilise la vue input.vProjects_Clean
    qui nettoie Lat/Lng/GLA et évite le type geography.
    """
    query = """
        SELECT TOP 1
            ProjectID,
            Name,
            City,
            YearBuilt,
            Lat,
            Lng,
            GLA
        FROM input.vProjects_Clean
        WHERE ProjectID = ?
    """
    df = read_sql(query, params=[project_id])

    if df.empty:
        raise ValueError(f"Aucun projet trouvé pour ProjectID={project_id} dans input.vProjects_Clean")

    row = df.iloc[0]

    # Vérif NaN
    missing = [c for c in FEATURE_COLS if pd.isna(row[c])]
    if missing:
        raise ValueError(
            f"Projet ProjectID={project_id} incomplet : {missing} manquant(s) "
            f"(Lat/Lng/GLA nécessaires pour le KNN)."
        )

    return row


def find_comparables(project_id: int, top_n: int = 5) -> pd.DataFrame:
    # 1) charger universe
    universe = load_training_universe()

    # 2) charger projet
    project = load_project(project_id)

    x_query = np.array([[float(project["Lat"]), float(project["Lng"]), float(project["GLA"])]])

    # 3) charger modèle
    model, office_ids = load_model()

    # 4) KNN
    distances, indices = model.kneighbors(x_query, n_neighbors=top_n)
    distances = distances[0]
    indices = indices[0]
    neighbor_ids = office_ids[indices]

    # 5) récupérer infos des voisins (dans l'ordre KNN)
    neighbors = universe[universe["OfficeID"].isin(neighbor_ids)].copy()

    # mapping rank/distance
    rank_map = {oid: rank for rank, oid in enumerate(neighbor_ids, start=1)}
    dist_map = {oid: dist for oid, dist in zip(neighbor_ids, distances)}
    neighbors["Rank"] = neighbors["OfficeID"].map(rank_map)
    neighbors["Distance"] = neighbors["OfficeID"].map(dist_map)
    neighbors["ProjectID"] = int(project_id)

    neighbors = neighbors.sort_values("Rank")

    # colonnes propres
    cols = [
        "ProjectID",
        "Rank",
        "OfficeID",
        "AssetName",
        "City",
        "Area",
        "GLA",
        "YearBuilt",
        "Lat",
        "Lng",
        "XOF_per_m2_per_year",
        "Distance",
    ]
    # Certaines colonnes peuvent être absentes selon ta vue : on filtre celles qui existent
    cols = [c for c in cols if c in neighbors.columns]

    return neighbors[cols]


if __name__ == "__main__":
    PROJECT_ID = 1   # change ici
    TOP_N = 5

    comps = find_comparables(PROJECT_ID, TOP_N)
    print("=== COMPARABLES ===")
    print(comps)
