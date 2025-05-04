import pandas as pd
df = pd.read_parquet("data/processed/cleanedData_sev.parquet")
test = df[df.Year == 2018]
print("Mean severity:", test.Cost_claims_year.mean())
print("Std  severity:", test.Cost_claims_year.std())
