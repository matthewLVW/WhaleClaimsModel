import pandas as pd

# 1. Load raw semicolon CSV
raw = pd.read_csv(
    "data/raw/Motor vehicle insurance data.csv",
    sep=";",
    dayfirst=True
)

# 2. Restrict to the severity subset
raw_sev = raw[raw["Cost_claims_year"] > 0]

# 3. Check each binary column
for col in ["Distribution_channel","Payment","Area","Second_driver"]:
    vals = raw_sev[col].unique()
    print(col, "unique values in raw_sev:", vals)
