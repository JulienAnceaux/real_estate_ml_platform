# src/db.py
import pyodbc
import pandas as pd

# TODO : remplace par ta vraie connection string (depuis test_sql_connection.py)
CONN_STR = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=server-cashflow.database.windows.net;"
    "Database=CashflowDB;"
    "UID=sqladmin;"
    "PWD=Test@2025!;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

def get_connection():
    return pyodbc.connect(CONN_STR)

def read_sql(query: str, params=None) -> pd.DataFrame:
    cnx = get_connection()
    try:
        df = pd.read_sql(query, cnx, params=params)
    finally:
        cnx.close()
    return df

def execute_sql(query: str, params=None):
    cnx = get_connection()
    try:
        cursor = cnx.cursor()
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        cnx.commit()
    finally:
        cnx.close()
