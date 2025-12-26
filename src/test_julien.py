from db import read_sql
df = read_sql("SELECT TOP 1 * FROM report.vTraining_Offices_ML")
print(df.columns)