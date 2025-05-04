import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load data
DATA = Path(__file__).parents[2] / "data/processed/cleanedData_sev.parquet"
df   = pd.read_parquet(DATA)

# 2. Split by Year
train = df[df.Year < 2018].copy()
test  = df[df.Year == 2018].copy()

# 3. Features & target (drop Exposure as a feature)
TARGET = "Cost_claims_year"
X_cols = [c for c in df.columns
          if c not in [TARGET, "Year", "Exposure"]]
X_train, y_train = train[X_cols], train[TARGET]
X_test,  y_test  = test [X_cols], test [TARGET]

# 4. Fit a Gamma‐log GLM weighted by exposure
glm = TweedieRegressor(power=2, alpha=0.0, link="log", max_iter=1000)
glm.fit(
    X_train,
    y_train,
    sample_weight=train["Exposure"]
)

# 5. Predict & evaluate
y_pred = glm.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred,
             sample_weight=test["Exposure"]))
print(f"Test RMSE (per‑year rate): {rmse:,.2f}")

# Empirical VaR99.5%
var_emp  = np.quantile(y_test, 0.995)
var_pred = np.quantile(y_pred, 0.995)
print(f"VaR 99.5 % true   : {var_emp:,.0f}")
print(f"VaR 99.5 % predict: {var_pred:,.0f}")

# 6. Coefficients
coefs = pd.Series(glm.coef_, index=X_cols).abs().sort_values(ascending=False)
print("\nTop 10 features by |coef|:")
print(coefs.head(10))

# 7. Plot top 10
top10 = coefs.head(10).sort_values()
plt.figure(figsize=(6,4))
top10.plot.barh()
plt.title("Top 10 GLM coefficients (abs), weighted by Exposure")
plt.tight_layout()
plt.savefig("figs/glm_coefs_weighted.png")
print("Saved coefficient plot → figs/glm_coefs_weighted.png")
