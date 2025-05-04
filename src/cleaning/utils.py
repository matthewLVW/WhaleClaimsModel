import pandas as pd, numpy as np, yaml, pathlib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

CFG = yaml.safe_load(
    (pathlib.Path(__file__).parent / "config.yaml").read_text()
)

def winsorize(df: pd.DataFrame, cols, lo=0.001, hi=0.999):
    for c in cols:
        lo_q, hi_q = df[c].quantile([lo, hi])
        df[c] = df[c].clip(lo_q, hi_q)
    return df

def calc_vif(df: pd.DataFrame, thresh: float):
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
