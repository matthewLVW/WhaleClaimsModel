import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
# 1. Load & prepare data
DATA     = Path(__file__).parents[2] / "data/processed/cleanedData_sev.parquet"
df       = pd.read_parquet(DATA)
train_df = df[df.Year < 2018].reset_index(drop=True)
test_df  = df[df.Year == 2018].reset_index(drop=True)

FEATURES = [c for c in df.columns if c not in ["Cost_claims_year", "Year"]]
TARGET   = "Cost_claims_year"

# 2. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_test  = scaler.transform(test_df[FEATURES])

y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

# 3. (Optional) further split for early‐stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# 4. Create LightGBM datasets
lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# 5. LightGBM & Tweedie parameters
params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "metric": "rmse",
    "verbose": -1,
}

# 6. Train with early stopping using callbacks
callbacks = [
    lgb.early_stopping(stopping_rounds=20),
    lgb.log_evaluation(period=50),
]
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=callbacks,
)

# 7. Predict & evaluate
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"LightGBM Tweedie RMSE: {rmse:,.2f}")

var_emp  = np.quantile(y_test, 0.995)
var_pred = np.quantile(y_pred, 0.995)
print(f"True VaR 99.5 %  : {var_emp:,.0f}")
print(f"LGBM VaR 99.5 % : {var_pred:,.0f}")

# 8. Feature importances
importances = pd.Series(gbm.feature_importance(importance_type="gain"),
                        index=FEATURES).sort_values(ascending=False)
print("\nTop 10 LGBM features by gain:")
print(importances.head(10))
