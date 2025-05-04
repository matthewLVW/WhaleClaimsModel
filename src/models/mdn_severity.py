import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# -- Hyperparameters --
K_MIX     = 5       # number of mixture components
HID       = 128     # hidden layer size
LR        = 1e-3
EPOCHS    = 200
BATCH     = 256
PATIENCE  = 20      # early stopping patience
VAL_SPLIT = 0.1

# 1. Load & prepare data
DATA = Path(__file__).parents[2] / "data/processed/cleanedData_sev.parquet"
df   = pd.read_parquet(DATA)
train_df = df[df.Year < 2018].reset_index(drop=True)
test_df  = df[df.Year == 2018].reset_index(drop=True)

FEATURES = [c for c in df.columns if c not in ["Cost_claims_year", "Year"]]
scaler   = StandardScaler()
X = scaler.fit_transform(train_df[FEATURES])
y = np.log1p(train_df["Cost_claims_year"].values)  # log-scale target

# Train/val split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, random_state=42
)

# DataLoaders
train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float().unsqueeze(-1))
val_ds   = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(-1))
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

# Prepare test
X_te = scaler.transform(test_df[FEATURES])
y_te = test_df["Cost_claims_year"].values

# 2. Define the MDN
class MDN(nn.Module):
    def __init__(self, in_dim, hid, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pi   = nn.Linear(hid, K)
        self.mu   = nn.Linear(hid, K)
        self.logs = nn.Linear(hid, K)

    def forward(self, x):
        h = self.net(x)
        pi    = F.softmax(self.pi(h), dim=-1)
        mu    = self.mu(h)
        sigma = F.softplus(self.logs(h)) + 1e-3
        return pi, mu, sigma

    def mdn_loss(self, pi, mu, sigma, y):
        # y: [B,1] -> [B,K]
        y_expand = y.expand_as(mu)
        gaussian = torch.exp(-0.5 * ((y_expand - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        prob     = (pi * gaussian).sum(dim=1) + 1e-8
        return -torch.log(prob).mean()

# 3. Train with early stopping & LR scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MDN(len(FEATURES), HID, K_MIX).to(device)
opt    = optim.Adam(model.parameters(), lr=LR)
sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
best_loss, epochs_no_improve = np.inf, 0

for ep in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pi, mu, sigma = model(xb)
        loss = model.mdn_loss(pi, mu, sigma, yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        opt.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pi, mu, sigma = model(xb)
            val_loss += model.mdn_loss(pi, mu, sigma, yb).item() * xb.size(0)
    val_loss /= len(val_ds)
    sched.step(val_loss)

    print(f"Epoch {ep} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "mdn_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping")
            break

# Load best model
model.load_state_dict(torch.load("mdn_best.pt"))

# 4. Evaluate on test (log-scale then back-transform)
model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_te).float().to(device)
    pi, mu, sigma = model(X_test_t)
    y_pred_log = (pi * mu).sum(dim=1).cpu().numpy()
y_pred = np.expm1(y_pred_log)

# Metrics
rmse = np.sqrt(((y_pred - y_te) ** 2).mean())
var_emp  = np.quantile(y_te, 0.995)
var_pred = np.quantile(y_pred, 0.995)
print(f"MDN RMSE: {rmse:.2f}")
print(f"True VaR99.5%: {var_emp:.0f} | MDN VaR99.5%: {var_pred:.0f}")
