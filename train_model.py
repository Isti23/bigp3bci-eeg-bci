import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, classification_report)
import matplotlib.pyplot as plt

# ── Konfigurasi ──────────────────────────────────────────────
BASE            = "/content/drive/MyDrive/bigP3BCI"
TRAIN_X         = f"{BASE}/Train_X_B_norm.npy"
TRAIN_Y         = f"{BASE}/Train_y_B_norm.npy"
TEST_X          = f"{BASE}/Test_X_B_norm.npy"
TEST_Y          = f"{BASE}/Test_y_B_norm.npy"
BEST_MODEL_PATH = f"{BASE}/best_conformer_B.pth"
CKPT_PATH       = f"{BASE}/checkpoint_conformer_B.pth"

BATCH_SIZE  = 256
EPOCHS      = 70
LR          = 1e-3
N_CHANNELS  = 32
N_TIMES     = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")



# ── Dataset ──────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)
        n_pos  = int((self.y == 1).sum())
        n_neg  = int((self.y == 0).sum())
        self.pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        print(f"   {len(self.y):,} epoch | T:{n_pos:,} NT:{n_neg:,} | "
              f"pos_weight:{self.pos_weight.item():.2f}")

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return x, y


# ── EEG-Conformer ────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, n_channels=32, emb_size=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=40, bias=False),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            nn.Dropout(0.5),
            nn.Conv2d(40, emb_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(emb_size), nn.ELU(),
        )

    def forward(self, x):
        x = self.conv(x).squeeze(2)
        return x.permute(0, 2, 1)  


class ConformerBlock(nn.Module):
    def __init__(self, emb_size=40, num_heads=4, ff_dim=128, dropout=0.5):
        super().__init__()
        self.attn  = nn.MultiheadAttention(emb_size, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(emb_size, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_size), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        return self.norm2(x + self.ff(x))


class EEGConformer(nn.Module):
    def __init__(self, n_channels=32, emb_size=40,
                 num_heads=4, num_layers=2, ff_dim=128, dropout=0.5):
        super().__init__()
        self.patch_emb  = PatchEmbedding(n_channels, emb_size)
        self.conformer  = nn.Sequential(
            *[ConformerBlock(emb_size, num_heads, ff_dim, dropout)
              for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(emb_size, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.conformer(x)
        return self.classifier(x.permute(0, 2, 1)).squeeze(1)


# ── Train & Eval ─────────────────────────────────────────────
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out  = model(X)
        loss = crit(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        preds.extend((torch.sigmoid(out) > 0.5).long().cpu().numpy())
        labels.extend(y.long().cpu().numpy())
        if (i+1) % 200 == 0:
            print(f"   Batch {i+1}/{len(loader)} | Loss:{loss.item():.4f}")
    return total_loss/len(loader), accuracy_score(labels, preds)


def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, probs, preds, labels = 0, [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            total_loss += crit(out, y).item()
            p = torch.sigmoid(out).cpu().numpy()
            probs.extend(p)
            preds.extend((p > 0.5).astype(int))
            labels.extend(y.long().cpu().numpy())
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds)
    return total_loss/len(loader), acc, auc, f1, labels, preds


# ── Main ─────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  EEG-Conformer — Study B — Cross-Session")
    print("="*55)

    print("\n📂 Loading dataset...")
    train_set = EEGDataset(TRAIN_X, TRAIN_Y)
    test_set  = EEGDataset(TEST_X,  TEST_Y)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n🧠 Inisialisasi EEG-Conformer...")
    model     = EEGConformer(N_CHANNELS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_set.pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)
    print(f"   Parameter: {sum(p.numel() for p in model.parameters()):,}")

    best_auc, best_epoch = 0.0, 0
    history     = {"train_loss":[], "train_acc":[], "test_loss":[],
                   "test_acc":[], "test_auc":[], "test_f1":[]}
    start_epoch = 1

    if os.path.exists(CKPT_PATH):
        print(f"\n🔄 Resume dari checkpoint...")
        ckpt = ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_auc    = ckpt["best_auc"]
        best_epoch  = ckpt["best_epoch"]
        history     = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"   Resume dari epoch {start_epoch} | best AUC: {best_auc:.4f}")

    # ── Training loop ────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch}/{EPOCHS} | LR:{scheduler.get_last_lr()[0]:.6f}")
        print(f"{'─'*50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc, test_auc, test_f1, _, _ = evaluate(
            model, test_loader, criterion, DEVICE)
        scheduler.step()

        for k, v in zip(["train_loss","train_acc","test_loss","test_acc","test_auc","test_f1"],
                         [train_loss, train_acc, test_loss, test_acc, test_auc, test_f1]):
            history[k].append(v)

        print(f"\n  Train → Loss:{train_loss:.4f} | Acc:{train_acc:.4f}")
        print(f"  Test  → Loss:{test_loss:.4f} | Acc:{test_acc:.4f} "
              f"| AUC:{test_auc:.4f} | F1:{test_f1:.4f}")

        if test_auc > best_auc:
            best_auc, best_epoch = test_auc, epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  🏆 Best model! AUC:{best_auc:.4f}")

        torch.save({"epoch":epoch, "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "history":history, "best_auc":best_auc,
                    "best_epoch":best_epoch}, CKPT_PATH)
        print(f"  💾 Checkpoint saved")

    # ── Evaluasi final ───────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  HASIL AKHIR — Best epoch: {best_epoch}")
    print(f"{'='*55}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    _, acc, auc, f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, DEVICE)

    print(f"\n  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['NonTarget','Target'])}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"],  label="Test")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["test_acc"],  label="Test")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[2].plot(history["test_auc"], label="AUC", color="green")
    axes[2].plot(history["test_f1"],  label="F1",  color="orange")
    axes[2].set_title("AUC & F1"); axes[2].legend()
    plt.tight_layout()
    plot_path = f"{BASE}/training_history_B.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n  📊 Plot      : {plot_path}")
    print(f"  💾 Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()