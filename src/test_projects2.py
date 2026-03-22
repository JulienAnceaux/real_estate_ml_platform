from db import read_sql

query = """
SELECT TOP 20
    ProjectID,
    Name,
    City,
    Lat,
    Lng,
    GLA
FROM input.vProjects_Clean
ORDER BY ProjectID
"""

df = read_sql(query)
print(df)

print("\n--- Missing values ---")
print(df[["Lat", "Lng", "GLA"]].isna().sum())