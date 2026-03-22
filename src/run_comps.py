# src/run_comps.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from db import read_sql, execute_sql, get_connection

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "knn_offices.joblib"
INDEX_PATH = MODELS_DIR / "knn_offices_index.npy"

FEATURE_COLS = ["Lat", "Lng"]


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

    x_query = np.array([[float(project["Lat"]), float(project["Lng"])]])

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

def insert_run(
    project_id: int, 
    top_n: int,
    market_version: int | None = None,
    ran_by : str | None = "python",
    note : str | None = "KNN run from run_comps.py",
) -> int:
    """
    Insère une ligne dans report.CompsRuns et renvoie le RunID créé.
    """
    query = """
        INSERT INTO report.CompsRuns 
        (
            ProjectID, 
            TopN,
            MarketVersion,
            RanBy,
            RanAtUtc,
            Note)
        OUTPUT INSERTED.RunID
        VALUES (?, ?, ?, ?, SYSUTCDATETIME(), ?)
    """

    result = execute_sql(
        query, 
        params=[project_id, top_n, market_version, ran_by, note], 
        fetch_one=True)

    if result is None:
        raise ValueError("Impossible de récupérer le RunID après insertion dans report.CompsRuns.")

    return int(result[0])


def insert_results(run_id: int, comps: pd.DataFrame) -> None:

    """
    Insère les comparables dans report.CompsRunsResults.
    """
    if comps.empty:
        print("Aucun comparable à insérer.")
        return

    query = """
        INSERT INTO report.CompsRunResults
        (
            RunID,
            OfficeID,
            AssetName,
            City,
            Lat,
            Lng,
            GLA,
            YearBuilt,
            Distance_km
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for _, row in comps.iterrows():
        execute_sql(
            query,
            params=[
                int(run_id),
                int(row["OfficeID"]),
                row.get("AssetName"),
                row.get("City"),
                float(row["Lat"]) if pd.notna(row.get("Lat")) else None,
                float(row["Lng"]) if pd.notna(row.get("Lng")) else None,
                float(row["GLA"]) if pd.notna(row.get("GLA")) else None,
                int(row["YearBuilt"]) if pd.notna(row.get("YearBuilt")) else None,
                float(row["Distance"]) if pd.notna(row.get("Distance")) else None,
            ],
        )


if __name__ == "__main__":
    
    sql_project = """
    SELECT TOP 1 ProjectID
    FROM input.Projects
    ORDER BY ProjectID DESC
    """
    PROJECT_ID = int(read_sql(sql_project).iloc[0]["ProjectID"])

    print(f"ProjectID utilisé : {PROJECT_ID}")
    
    TOP_N = 8

    comps = find_comparables(PROJECT_ID, TOP_N)
    run_id = insert_run(PROJECT_ID, TOP_N)
    insert_results(run_id, comps)

    print(f"RunID créé : {run_id}")
    print(comps)
