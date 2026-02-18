from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db import read_sql

MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
KNN_PATH = MODELS_DIR / "knn_offices.joblib"
PREP_PATH = MODELS_DIR / "preprocessor_offices.joblib"
REF_PATH = MODELS_DIR / "knn_offices_ref.parquet"

SQL_QUERY = "SELECT * FROM report.vTraining_Offices_ML"

ID_COL = "OfficeID"
NUM_FEATURES = ["Lat", "Lng", "GLA", "YearBuilt"]
CAT_FEATURES = ["City", "Area"]

REF_COLS = [ID_COL, "AssetName", "City", "Area", "GLA", "YearBuilt", "Lat", "Lng", "XOF_per_m2_per_year"]

def main():
    df = read_sql(SQL_QUERY)
    if df is None or df.empty:
        raise RuntimeError("Aucune donnée retournée")

    # clean
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES).copy()
    df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)
    df[ID_COL] = df[ID_COL].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(df)

    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(X)

    joblib.dump(knn, KNN_PATH)
    joblib.dump(preprocessor, PREP_PATH)

    df[REF_COLS].reset_index(drop=True).to_parquet(REF_PATH, index=False)

    print("✅ saved:", KNN_PATH)
    print("✅ saved:", PREP_PATH)
    print("✅ saved:", REF_PATH)
    print("rows:", len(df))

if __name__ == "__main__":
    main()
