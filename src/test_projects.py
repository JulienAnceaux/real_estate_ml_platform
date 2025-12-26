from db import read_sql

# 1) est-ce que la table/vue existe ?
print(read_sql("SELECT TOP 5 * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE '%Project%'"))

# 2) test direct sur input.Projects
df = read_sql("SELECT TOP 5 * FROM input.Projects")
print(df.columns)
print(df.head())
