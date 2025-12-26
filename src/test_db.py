from db import read_sql

df = read_sql("SELECT TOP 5 * FROM report.vTraining_Offices_ML")
print(df)
