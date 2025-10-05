import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# =============================
# Config (kept to your spec)
# =============================
CSV_PATH = "energy_maps_with_labels_g4.csv"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 120               # give it a bit more runway
BASE_LR = 8e-4
WEIGHT_DECAY = 1e-5

# Detector/grid
N_PIX = 8
MIN_EDGE = -25.0
MAX_EDGE = +25.0
PIXEL_W = (MAX_EDGE - MIN_EDGE) / N_PIX

# Model
D_MODEL = 128
NHEAD = 8
NLAYERS = 4
FF_DIM = 256
DROPOUT = 0.1
USE_INPUT_NORMALIZATION = True  # matches your CNN/MLP centroid setup

# Loss weights
W_CE = 1.0           # cell classification
W_OFF = 2.0          # in-cell offset regression
W_ABS = 0.5          # small absolute (x,y) regression for stabilization

LABEL_SMOOTH = 0.05  # label smoothing for CE; helps generalization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Data
# =============================
df = pd.read_csv(CSV_PATH)
X = df.iloc[:, :-2].values.reshape(-1, 1, N_PIX, N_PIX).astype(np.float32)
y_abs = df[["x_true", "y_true"]].values.astype(np.float32)  # absolute coords in physical units

if USE_INPUT_NORMALIZATION:
    X = X / (X.sum(axis=(2, 3), keepdims=True) + 1e-8)

def coords_to_cell_and_offset(xy):
    """
    Convert absolute (x,y) to:
      - cell index in [0..63] (row-major: y,x)
      - in-cell offsets normalized to [-0.5, 0.5] along each axis.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    # Clamp to detector bounds
    x = np.clip(x, MIN_EDGE + 1e-6, MAX_EDGE - 1e-6)
    y = np.clip(y, MIN_EDGE + 1e-6, MAX_EDGE - 1e-6)

    # cell indices along x/y (0..7)
    ix = np.floor((x - MIN_EDGE) / PIXEL_W).astype(np.int64)
    iy = np.floor((y - MIN_EDGE) / PIXEL_W).astype(np.int64)
    ix = np.clip(ix, 0, N_PIX-1)
    iy = np.clip(iy, 0, N_PIX-1)

    # centers
    cx = MIN_EDGE + (ix + 0.5) * PIXEL_W
    cy = MIN_EDGE + (iy + 0.5) * PIXEL_W

    # offsets normalized by pixel width -> roughly in [-0.5, 0.5]
    offx = (x - cx) / PIXEL_W
    offy = (y - cy) / PIXEL_W
    off = np.stack([offx, offy], axis=1).astype(np.float32)

    cell_id = (iy * N_PIX + ix).astype(np.int64)  # row-major
    return cell_id, off

cell_id_all, off_all = coords_to_cell_and_offset(y_abs)

X_train, X_test, y_abs_train, y_abs_test, cid_train, cid_test, off_train, off_test = train_test_split(
    X, y_abs, cell_id_all, off_all, test_size=0.2, random_state=SEED
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_abs_train),
                         torch.tensor(cid_train), torch.tensor(off_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_abs_test),
                         torch.tensor(cid_test), torch.tensor(off_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# =============================
# Helpers: grid geometry
# =============================
centers_1d = np.linspace(MIN_EDGE + PIXEL_W/2, MAX_EDGE - PIXEL_W/2, N_PIX, dtype=np.float32)
xx_np, yy_np = np.meshgrid(centers_1d, centers_1d, indexing="xy")
xx_flat = xx_np.reshape(-1)  # (64,)
yy_flat = yy_np.reshape(-1)  # (64,)
xx_t = torch.tensor(xx_np, dtype=torch.float32, device=DEVICE).view(1, 1, N_PIX, N_PIX)
yy_t = torch.tensor(yy_np, dtype=torch.float32, device=DEVICE).view(1, 1, N_PIX, N_PIX)

# =============================
# Model
# =============================
class HybridCellOffset(nn.Module):
    """
    Transformer encoder over 64 tokens.
    Heads:
      - cell logits over 64 tokens (classification)
      - per-token offset (2-dim), we select offsets at the GT/pred cell
      - small absolute (x,y) head from [CLS] for stabilization (optional)
    """
    def __init__(self):
        super().__init__()
        d_model = D_MODEL
        # CNN stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, d_model//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, N_PIX*N_PIX, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=NHEAD, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=NLAYERS)

        # Heads
        self.head_cell = nn.Sequential(  # -> (B,64)
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.head_off = nn.Sequential(   # -> (B,64,2)
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )
        self.head_abs = nn.Sequential(   # -> (B,2)
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x):
        B = x.size(0)
        feats = self.stem(x)                     # (B,d,8,8)
        tok = feats.flatten(2).transpose(1,2)    # (B,64,d)
        tok = tok + self.pos
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)       # (B,65,d)
        h = self.enc(seq)
        h_cls = h[:, 0, :]
        h_tok = h[:, 1:, :]                      # (B,64,d)

        cell_logits = self.head_cell(h_tok).squeeze(-1)  # (B,64)
        offsets = self.head_off(h_tok)                  # (B,64,2) in [-?,?], learned
        abs_xy = self.head_abs(h_cls)                   # (B,2)

        return cell_logits, offsets, abs_xy

# =============================
# Losses
# =============================
class LabelSmoothedCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        """
        logits: (B, C), target: (B,) int64
        """
        n_class = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-true_dist * log_probs).sum(dim=1).mean()

def gather_offsets(offsets, cell_idx):
    """
    offsets: (B, 64, 2)
    cell_idx: (B,) int64
    Returns selected offsets: (B,2)
    """
    B = offsets.size(0)
    idx = cell_idx.view(B, 1, 1).expand(B, 1, 2)  # (B,1,2)
    gathered = torch.gather(offsets, 1, idx)      # (B,1,2)
    return gathered.squeeze(1)                    # (B,2)

def cell_to_center_xy(cell_idx):
    """
    cell_idx: (B,) int64
    Returns centers (B,2) in absolute coords
    """
    # convert flat idx -> (iy, ix)
    iy = cell_idx // N_PIX
    ix = cell_idx % N_PIX
    cx = MIN_EDGE + (ix.to(torch.float32) + 0.5) * PIXEL_W
    cy = MIN_EDGE + (iy.to(torch.float32) + 0.5) * PIXEL_W
    return torch.stack([cx, cy], dim=1)

# =============================
# Train / Eval
# =============================
def train_and_eval():
    model = HybridCellOffset().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    # Cosine schedule with linear warmup (5 epochs)
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        # cosine from warmup..EPOCHS
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ce_loss = LabelSmoothedCELoss(LABEL_SMOOTH)
    l1 = nn.L1Loss()

    history = {"train": [], "val": [], "mae": [], "dist": []}

    for epoch in range(1, EPOCHS+1):
        # ---- Train ----
        model.train()
        total_train = 0.0
        for xb, y_abs_b, cid_b, off_b in train_loader:
            xb = xb.to(DEVICE)
            y_abs_b = y_abs_b.to(DEVICE)
            cid_b = cid_b.to(DEVICE)
            off_b = off_b.to(DEVICE)

            optimizer.zero_grad()
            cell_logits, offsets_all, abs_xy = model(xb)

            # select offsets at GT cell
            off_sel = gather_offsets(offsets_all, cid_b)        # (B,2)
            # reconstruct predicted abs coords from GT cell + offset
            center_xy = cell_to_center_xy(cid_b)                # (B,2)
            pred_abs_from_off = center_xy + off_sel * PIXEL_W   # (B,2)

            loss = (
                W_CE  * ce_loss(cell_logits, cid_b) +
                W_OFF * l1(off_sel, off_b) +
                W_ABS * l1(pred_abs_from_off, y_abs_b)  # stabilize scale
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += loss.item() * xb.size(0)

        scheduler.step()
        avg_train = total_train / len(train_loader.dataset)

        # ---- Val ----
        model.eval()
        val_loss, mae, dist = 0.0, 0.0, 0.0
        with torch.no_grad():
            for xb, y_abs_b, cid_b, off_b in test_loader:
                xb = xb.to(DEVICE)
                y_abs_b = y_abs_b.to(DEVICE)
                cid_b = cid_b.to(DEVICE)
                off_b = off_b.to(DEVICE)

                cell_logits, offsets_all, abs_xy = model(xb)
                # pick the predicted cell
                pred_cid = torch.argmax(cell_logits, dim=1)
                pred_off = gather_offsets(offsets_all, pred_cid)
                pred_center = cell_to_center_xy(pred_cid)
                pred_abs = pred_center + pred_off * PIXEL_W

                # Val loss mirrors train (but uses GT for offsets for stability)
                off_sel = gather_offsets(offsets_all, cid_b)
                center_xy = cell_to_center_xy(cid_b)
                pred_abs_from_off = center_xy + off_sel * PIXEL_W
                loss = (
                    W_CE  * ce_loss(cell_logits, cid_b) +
                    W_OFF * l1(off_sel, off_b) +
                    W_ABS * l1(pred_abs_from_off, y_abs_b)
                )

                val_loss += loss.item() * xb.size(0)
                mae += torch.mean(torch.abs(pred_abs - y_abs_b), dim=1).sum().item()
                dist += torch.norm(pred_abs - y_abs_b, dim=1).sum().item()

        val_loss /= len(test_loader.dataset)
        mae /= len(test_loader.dataset)
        dist /= len(test_loader.dataset)

        history["train"].append(avg_train)
        history["val"].append(val_loss)
        history["mae"].append(mae)
        history["dist"].append(dist)

        print(f"Epoch {epoch:3d}: Train Loss={avg_train:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.3f}, Dist Err={dist:.3f}, LR={scheduler.get_last_lr()[0]:.2e}")

    # ---- Curves ----
    plt.figure()
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"], label="Val Loss")
    plt.plot(history["mae"], label="Val MAE")
    plt.plot(history["dist"], label="Val Dist Err")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title("Hybrid Transformer (cell + offset)"); plt.show()

    # ---- Final eval ----
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, y_abs_b, cid_b, off_b in test_loader:
            xb = xb.to(DEVICE)
            y_abs_b = y_abs_b.to(DEVICE)

            cell_logits, offsets_all, _ = model(xb)
            pred_cid = torch.argmax(cell_logits, dim=1)
            pred_off = gather_offsets(offsets_all, pred_cid)
            pred_center = cell_to_center_xy(pred_cid)
            pred_abs = pred_center + pred_off * PIXEL_W
            preds.append(pred_abs.cpu().numpy())
            ys.append(y_abs_b.cpu().numpy())

    preds = np.vstack(preds)
    ys = np.vstack(ys)
    mean_dist = np.linalg.norm(preds - ys, axis=1).mean()
    print(f"hybrid mean distance error: {mean_dist:.4f}")

    # ---- Centroid baseline (same as before; per-sample normalized maps) ----
    energy_maps = X_test.squeeze(1)   # (N,8,8)
    x_num = (energy_maps * xx_np).sum(axis=(1, 2))
    y_num = (energy_maps * yy_np).sum(axis=(1, 2))
    E_sum = energy_maps.sum(axis=(1, 2)) + 1e-8
    y_c = np.column_stack((x_num/E_sum, y_num/E_sum))
    centroid_mean_dist = np.linalg.norm(y_abs_test - y_c, axis=1).mean()
    print(f"Centroid mean distance error:    {centroid_mean_dist:.4f}")

if __name__ == "__main__":
    train_and_eval()