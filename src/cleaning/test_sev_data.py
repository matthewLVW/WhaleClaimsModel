# src/cleaning/test_sev_data.py
import pandas as pd
from pathlib import Path

# 1. Paths
SEV_PATH = Path(__file__).parents[2] / "data/processed/cleanedData_sev.parquet"

# 2. Load
df = pd.read_parquet(SEV_PATH)

# 3. Expected columns
expected_base = {
    "Exposure", "Cost_claims_year", "Year",
    "Distribution_channel", "Payment", "Area", "Second_driver",
    "Length", "Weight", "Power", "Cylinder_capacity"
}
missing_expected = expected_base - set(df.columns)

print("➤ Rows:", df.shape[0])
print("➤ Columns:", df.shape[1])
print("\n❗ Missing expected columns:", missing_expected or "None")

# 4. Missingness
miss = df.isna().sum()
miss_pos = miss[miss > 0]
if miss_pos.empty:
    print("\n❗ Columns with missing values: None")
else:
    print("\n❗ Columns with missing values:\n", miss_pos)

# 5. Numeric feature stats
repr_features = ["Length", "Weight", "Power", "log_Premium", "log_Value_vehicle"]
print("\nℹ️ Sample numeric stats:\n", df[repr_features].describe().T[["mean","std","min","max"]])

# 6. Binary columns check
for col in ["Distribution_channel", "Payment", "Area", "Second_driver"]:
    vals = df[col].unique() if col in df.columns else "Missing"
    print(f"\nℹ️ Unique values in {col}:", vals)

# 7. Head sample
print("\n🔍 Sample rows:\n", df.head())
