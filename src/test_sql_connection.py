import pyodbc
import pandas as pd

conn_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=server-cashflow.database.windows.net;"
    "DATABASE=CashflowDB;"
    "UID=sqladmin;"
    "PWD=Test@2025!;"
    "TrustServerCertificate=yes;"
)

cnx = pyodbc.connect(conn_str)

query = "SELECT TOP 5 * FROM report.vTraining_Offices_ML;"
df = pd.read_sql(query, cnx)
print(df.head())

