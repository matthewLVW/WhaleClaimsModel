"""
Lightweight EDA: prints NA counts, skew, and high correlations (>0.9).
Usage: python -m src.cleaning.00_eda
"""
import pandas as pd, pathlib, numpy as np
RAW = pathlib.Path(__file__).parents[2] / "data/raw/Motor vehicle insurance data.csv"

df = pd.read_csv(
        RAW,
        sep=";",              # keep the semicolon
        parse_dates=[
            "Date_start_contract",
            "Date_last_renewal",
            "Date_next_renewal",
            "Date_birth",
            "Date_driving_licence",
            "Date_lapse"
        ],
)

print("\n--- Missing % per column ---")
print(df.isna().mean().sort_values(ascending=False).head(35))

print("\n--- Numeric skew (>2 only) ---")
num = df.select_dtypes(include=np.number)
sk = num.skew().sort_values(ascending=False)
print(sk[sk > 2].head(10))

print("\n--- High Pearson correlations (>0.9) ---")
corr = num.corr().abs()
high = np.where(np.triu(corr, 1) > 0.9)
for i, j in zip(*high):
    print(f"{corr.columns[i]}  vs  {corr.columns[j]}  =  {corr.iloc[i,j]:.2f}")
