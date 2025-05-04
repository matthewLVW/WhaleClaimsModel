# Motor Insurance Severity Modeling

I put together an end‑to‑end proof‑of‑concept that tackles motor‑insurance claim severity. Here’s what I did:

1. **Data‑Prep**  
   - Read in a semicolon‑delimited CSV, engineered `Exposure` and ages  
   - Winsorized extreme values, log‑transformed skewed features  
   - One‑hot and binary‑encoded categorical columns, scaled numerics  
   - Pruned collinearity via VIF  
   - Wrote out two Parquets:  
     - `data/processed/cleanedData_full.parquet` (all policies)  
     - `data/processed/cleanedData_sev.parquet` (only positive claims)

2. **Baseline Models**  
   - **Gamma GLM** (log‑link with exposure offset) → RMSE ≈ €1 467  
   - **LightGBM (Tweedie)** → RMSE ≈ €1 220  
   - Those errors are about **3×** the mean claim (€632), which really shows how tricky insurance tails are.

3. **Advanced Prototypes**  
   - **Mixture Density Network (MDN)** on log‑scale → RMSE ≈ €1 423, better tail capture  
   - This sets the stage for full **Tail‑Aware Flows** or **EVT‑penalized** nets.

4. **Conclusion**  
   - On their own, these models aren’t production‑grade—their global error and VaR estimates are still far off.  
   - Each approach does excel in different areas: GLM for interpretability, LightGBM for overall RMSE, MDN for tail behavior.  
   - Moving forward, I’m excited to explore **model layering** (e.g. stacking a flow over LightGBM residuals or blending a quantile net on MDN outputs) to combine their strengths into a hybrid, high‑accuracy severity framework.

---

## How to reproduce

```bash
git clone <this‑repo>
cd TailAwareFlows
pip install -r requirements.txt

# 1. Clean and prep the data
python -m src.cleaning.data_prep

# 2. Run the GLM baseline
python -m src.models.glm_severity

# 3. Run the LightGBM baseline
python -m src.models.lgbm_severity

# 4. Run the MDN prototype
python -m src.models.mdn_severity
