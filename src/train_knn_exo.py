from pathlib import Path
import pandas as pd
from db import read_sql


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SQL_QUERY = "SELECT * FROM report.vTraining_Offices_ML"


def load_training_df():
    df = read_sql(SQL_QUERY)

    if df is None or df.empty :
        raise RuntimeError("La requête SQL n'a retourné aucune donnée")
    
    return df

if __name__ == "__main__":
    df = load_training_df()
    print("SHAPE:", df.shape)
    print("COLUMNS", df.columns.tolist())
    print(df.head(10))

