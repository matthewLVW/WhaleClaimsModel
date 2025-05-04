"""
Data‑prep pipeline for the semicolon‑delimited motor‑portfolio file.
Creates TWO parquet files:
    • cleanedData_full.parquet   – all rows (0‑claim + positive)
    • cleanedData_sev.parquet    – only rows with Cost_claims_year > 0
"""

import pandas as pd, numpy as np, pathlib, yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --------------------------------------------------------------------
# 1.  Paths & YAML config
# --------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW  = ROOT / "data/raw/Motor vehicle insurance data.csv"            # semicolon CSV
PROC_FULL = ROOT / "data/processed/cleanedData_full.parquet"
PROC_SEV  = ROOT / "data/processed/cleanedData_sev.parquet"

CFG = yaml.safe_load((pathlib.Path(__file__).parent / "config.yaml").read_text())

# --------------------------------------------------------------------
# 2.  Helper utilities
# --------------------------------------------------------------------
def winsorize(df, cols, low_quantile=0.001, high_quantile=0.999):
    for c in cols:
        lo, hi = df[c].quantile([low_quantile, high_quantile])
        df[c] = df[c].clip(lo, hi)
    return df


def remove_vif(df, thresh):
    keep = df.columns.tolist()
    while True:
        vifs = pd.Series(
            [variance_inflation_factor(df[keep].values, i) for i in range(len(keep))],
            index=keep,
        )
        worst = vifs.idxmax()
        if vifs.max() < thresh:
            break
        keep.remove(worst)
    return df[keep]

# --------------------------------------------------------------------
# 3.  Load raw CSV  (semicolon separator!)
# --------------------------------------------------------------------
def load_raw():
    #   decimal uses dot already (e.g., 216.99) so no decimal= needed
    date_cols = list(CFG["date_cols"].values())
    df = pd.read_csv(
    RAW,
    sep=";",
    parse_dates=date_cols,
    dayfirst=True        # <— tells pandas these are DD/MM/YYYY dates
)
    # derive Year & engineered ages
    df["Year"] = df[CFG["date_cols"]["start"]].dt.year
    df["Insured_Age"] = df["Year"] - df[CFG["date_cols"]["birth"]].dt.year
    df["Years_Licenced"] = df["Year"] - df[CFG["date_cols"]["licence"]].dt.year
    df["Vehicle_Age"] = df["Year"] - df["Year_matriculation"]
    return df

# --------------------------------------------------------------------
# 4.  Transform: winsorize, log‑skewed, encode, scale, VIF prune
# --------------------------------------------------------------------
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # 4.1  Winsorize the heavy‑tailed features
    skew_cols = CFG["numeric_cols"]["skew_log"]
    w_lo      = CFG["winsor"]["low_quantile"]
    w_hi      = CFG["winsor"]["high_quantile"]
    df = winsorize(df, skew_cols, low_quantile=w_lo, high_quantile=w_hi)

    # 4.2  Log‑transform the same skewed features
    for c in skew_cols:
        df[f"log_{c}"] = np.log1p(df[c])

    # 4.3  Build encoding + scaling pipeline
    cat_binary = CFG["categorical_cols"]["binary"]
    cat_onehot = CFG["categorical_cols"]["onehot"]
    num_std    = CFG["numeric_cols"]["standard"] + [f"log_{c}" for c in skew_cols]
    ct = ColumnTransformer(
    transformers=[
        ("bin",   "passthrough",                     cat_binary),
        ("cat",   OneHotEncoder(drop="first", sparse_output=False), cat_onehot),
        ("num",   StandardScaler(),                  num_std),
    ],
    remainder="drop",
)

    all_inputs =  cat_onehot + num_std + cat_binary
    arr = ct.fit_transform(df[all_inputs])

    feat_names = (
        cat_binary
        + ct.named_transformers_["cat"].get_feature_names_out(cat_onehot).tolist()
        + num_std
    )
    out = pd.DataFrame(arr, columns=feat_names, index=df.index)
    out[["Exposure","Cost_claims_year","Year"]] = df[["Exposure","Cost_claims_year","Year"]]
    # 4.4  Bring along the passthrough columns
    keep_passthrough = ["Exposure", "Cost_claims_year", "Year"]
    out[keep_passthrough] = df[keep_passthrough]

    # 4.5  Clean Inf/NaN before feature‐selection
    out = out.replace([np.inf, -np.inf], np.nan)

    # --- A) EXCLUDE target & non‑predictors from VIF ---
    feature_cols = [c for c in out.columns if c not in keep_passthrough]

    # --- B) IMPUTE NaNs in features before VIF (median) ---
    #   median imputation is safer than fillna(0) for scaled data
    medians = out[feature_cols].median()
    out[feature_cols] = out[feature_cols].fillna(medians)

    # 4.6  Remove near‑constant features on the feature set
    mask = VarianceThreshold(1e-4).fit(out[feature_cols]).get_support()
    feature_pruned = out[feature_cols].loc[:, mask]

    # 4.7  VIF pruning on features only
    feature_pruned = remove_vif(feature_pruned, CFG["vif_threshold"])

    # 4.8  Re‑assemble final DataFrame: pruned features + passthrough
    out = pd.concat([feature_pruned, out[keep_passthrough]], axis=1)

    return out

# --------------------------------------------------------------------
# 5.  Main
# --------------------------------------------------------------------
def main():
    df_raw = load_raw()
    # create Exposure (pro‑rata on‑risk duration)
    df_raw["Exposure"] = (
        df_raw[CFG["date_cols"]["next"]] - df_raw[CFG["date_cols"]["last"]]
    ).dt.days.div(365).clip(upper=1.0)

    clean = transform(df_raw)

    PROC_FULL.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(PROC_FULL)
    clean[clean["Cost_claims_year"] > 0].to_parquet(PROC_SEV)
    print("✅  Full dataset  →", PROC_FULL)
    print("✅  Severity only →", PROC_SEV)

if __name__ == "__main__":
    main()
