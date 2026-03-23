"""
bci_decoding.py
===============
BCI Decoding untuk Study B — Checkerboard Paradigm
Menggunakan model EEG-Conformer yang sudah dilatih.

Cara pakai (di Mac lokal):
    python3 bci_decoding.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import mne

mne.set_log_level("WARNING")

# ── Konfigurasi ──────────────────────────────────────────────
BASE       = "/Users/istiqomah/Dev/Kuliah smt4/IDSC/bigp3bci-an-open-diverse-and-machine-learning-ready-p300-based-brain-computer-interface-dataset-1.0.0"
MODEL_PATH = f"{BASE}/best_conformer_B.pth"
STUDY_DIR  = f"{BASE}/bigP3BCI-data/StudyB"
PREP_DIR   = f"{BASE}/preprocessed_B"   # dipakai untuk hitung mean/std (Volt)
N_COLS     = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUBJECT_LAST_SESSION = {
    "B_01": "SE008", "B_02": "SE002", "B_03": "SE006",
    "B_04": "SE006", "B_06": "SE003", "B_10": "SE006",
    "B_11": "SE004", "B_12": "SE012", "B_13": "SE003",
    "B_14": "SE004", "B_17": "SE003", "B_18": "SE004",
    "B_20": "SE002"
}

SFREQ_NEW = 128
L_FREQ    = 0.5
H_FREQ    = 40.0
TMIN      = -0.2
TMAX      = 0.8

EEG_CHANNELS = [
    "EEG_F3",  "EEG_Fz",  "EEG_F4",  "EEG_T7",  "EEG_C3",  "EEG_Cz",
    "EEG_C4",  "EEG_T8",  "EEG_CP3", "EEG_CP4", "EEG_P3",  "EEG_Pz",
    "EEG_P4",  "EEG_PO7", "EEG_PO8", "EEG_Oz",  "EEG_FP1", "EEG_FP2",
    "EEG_F7",  "EEG_F8",  "EEG_FC5", "EEG_FC1", "EEG_FC2", "EEG_FC6",
    "EEG_CPz", "EEG_P7",  "EEG_P5",  "EEG_PO3", "EEG_POz", "EEG_PO4",
    "EEG_O1",  "EEG_O2"
]


# ── Model ────────────────────────────────────────────────────
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
        self.attn  = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(emb_size, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_size), nn.Dropout(dropout))
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
              for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(emb_size, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, x):
        x = self.patch_emb(x)
        x = self.conformer(x)
        return self.classifier(x.permute(0, 2, 1)).squeeze(1)


# ── Mapping char_channel index → CurrentTarget number ────────
def build_charidx_to_target(char_channels, n_cols=6):
    mapping = {}
    for idx, ch in enumerate(char_channels):
        parts = ch.split('_')
        row = int(parts[-2])
        col = int(parts[-1])
        mapping[idx] = (row - 1) * n_cols + col
    return mapping


# ── Load stats per subjek dari preprocessed_B (Volt) ─────────
def load_subject_stats(subject, prep_dir):
    files = [
        f for f in os.listdir(prep_dir)
        if f.startswith(subject) and "__SE001__Train__" in f
    ]
    if not files:
        return None, None

    all_epochs = []
    for fname in files:
        d = np.load(os.path.join(prep_dir, fname))
        all_epochs.append(d["epochs"])
        d.close()

    X    = np.concatenate(all_epochs, axis=0)
    mean = X.mean(axis=(0, 2), keepdims=True)  # (1, 32, 1)
    std  = X.std(axis=(0, 2),  keepdims=True) + 1e-8
    return mean, std


# ── Load & epoch satu file EDF ───────────────────────────────
def load_and_epoch(filepath, subject_mean, subject_std):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    char_channels = sorted([
        ch for ch in raw.ch_names
        if '_' in ch and len(ch.split('_')) == 3
    ])
    if len(char_channels) != 36:
        raise ValueError(f"Karakter channel tidak 36: {len(char_channels)}")

    charidx_to_target = build_charidx_to_target(char_channels, N_COLS)

    stim_begin  = raw.get_data(picks=["StimulusBegin"])[0]
    current_tgt = raw.get_data(picks=["CurrentTarget"])[0]
    char_data   = raw.get_data(picks=char_channels)
    sfreq_orig  = raw.info["sfreq"]

    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available)
    raw.set_channel_types({ch: "eeg" for ch in available})
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.resample(SFREQ_NEW, verbose=False)
    eeg_data = raw.get_data()  # Volt

    onset_orig = np.where(stim_begin > 0)[0]
    onset_new  = (onset_orig / sfreq_orig * SFREQ_NEW).astype(int)

    n_baseline = int(abs(TMIN) * SFREQ_NEW)
    n_after    = int(TMAX * SFREQ_NEW) + 1

    epochs_list, active_targets_list, targets_list = [], [], []

    for o_orig, o_new in zip(onset_orig, onset_new):
        start, end = o_new - n_baseline, o_new + n_after
        if start < 0 or end > eeg_data.shape[1]:
            continue

        epoch = eeg_data[:, start:end].copy().astype(np.float32)
        # Baseline correction
        epoch -= epoch[:, :n_baseline].mean(axis=1, keepdims=True)
        # Pad ke 32 channel
        if epoch.shape[0] < 32:
            pad   = np.zeros((32 - epoch.shape[0], epoch.shape[1]), dtype=np.float32)
            epoch = np.vstack([epoch, pad])
        # Pastikan shape (32, 128)
        epoch = epoch[:32, :128]
        if epoch.shape[1] < 128:
            epoch = np.pad(epoch, ((0, 0), (0, 128 - epoch.shape[1])))

        # Normalisasi dengan stats dari preprocessed_B (Volt)
        if subject_mean is not None:
            epoch = (epoch - subject_mean[0]) / subject_std[0]

        active_char_indices = np.where(char_data[:, o_orig] > 0)[0].tolist()
        active_tgt_nums = [charidx_to_target[i] for i in active_char_indices
                           if i in charidx_to_target]

        epochs_list.append(epoch)
        active_targets_list.append(active_tgt_nums)
        targets_list.append(int(current_tgt[o_orig]))

    epochs_arr = np.stack(epochs_list, axis=0) if epochs_list \
                 else np.zeros((0, 32, 128))
    return epochs_arr, active_targets_list, targets_list, charidx_to_target


# ── Decode satu file ─────────────────────────────────────────
def decode_file(filepath, model, subject_mean, subject_std, device, batch_size=256):
    epochs, active_targets_list, targets_list, charidx_to_target = \
        load_and_epoch(filepath, subject_mean, subject_std)

    if len(epochs) == 0:
        return [], []

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(epochs), batch_size):
            batch = torch.tensor(epochs[i:i+batch_size],
                                 dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
    probs = np.array(probs)

    all_target_nums = sorted(set(charidx_to_target.values()))
    tgt_to_idx = {t: i for i, t in enumerate(all_target_nums)}
    n_chars = len(all_target_nums)

    # Aggregate per trial
    trials = []
    current_trial = None
    for p, active_tgts, tgt in zip(probs, active_targets_list, targets_list):
        if current_trial is None or current_trial["target"] != tgt:
            current_trial = {"scores": np.zeros(n_chars), "target": tgt}
            trials.append(current_trial)
        for tgt_num in active_tgts:
            if tgt_num in tgt_to_idx:
                current_trial["scores"][tgt_to_idx[tgt_num]] += p

    y_true, y_pred = [], []
    for trial in trials:
        best_idx      = int(np.argmax(trial["scores"]))
        predicted_tgt = all_target_nums[best_idx]
        y_true.append(trial["target"])
        y_pred.append(predicted_tgt)

    return y_true, y_pred


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BCI DECODING — Study B — Cross-Session")
    print("=" * 60)

    print(f"\n🧠 Loading model...")
    model = EEGConformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=False))
    model.eval()
    print(f"   ✅ Model loaded | Device: {DEVICE}")

    all_true, all_pred = [], []
    results = []

    for subject, last_session in SUBJECT_LAST_SESSION.items():
        print(f"\n{'─'*55}")
        print(f"  {subject} — {last_session}")

        mean, std = load_subject_stats(subject, PREP_DIR)
        if mean is None:
            print(f"  ⚠️  Stats tidak ditemukan, skip!")
            continue

        test_dir = os.path.join(STUDY_DIR, subject, last_session, "Test", "CB")
        if not os.path.exists(test_dir):
            print(f"  ⚠️  Folder tidak ditemukan: {test_dir}")
            continue

        test_files = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir) if f.endswith(".edf")
        ])

        subj_true, subj_pred = [], []
        for fpath in test_files:
            try:
                y_true, y_pred = decode_file(fpath, model, mean, std, DEVICE)
                if not y_true:
                    continue
                subj_true.extend(y_true)
                subj_pred.extend(y_pred)
                n_correct = sum(t == p for t, p in zip(y_true, y_pred))
                print(f"  📄 {os.path.basename(fpath)}: "
                      f"{n_correct}/{len(y_true)} benar "
                      f"({n_correct/len(y_true)*100:.0f}%)")
            except Exception as e:
                print(f"  ❌ {os.path.basename(fpath)}: {e}")

        if subj_true:
            n_correct = sum(t == p for t, p in zip(subj_true, subj_pred))
            acc = n_correct / len(subj_true)
            print(f"  ✅ {subject}: {acc:.1%} ({n_correct}/{len(subj_true)})")
            all_true.extend(subj_true)
            all_pred.extend(subj_pred)
            results.append({
                "subject":  subject,
                "session":  last_session,
                "correct":  n_correct,
                "total":    len(subj_true),
                "accuracy": acc
            })

    print(f"\n{'='*60}")
    print(f"  HASIL AKHIR BCI DECODING")
    print(f"{'='*60}")

    if all_true:
        n_correct   = sum(t == p for t, p in zip(all_true, all_pred))
        overall_acc = n_correct / len(all_true)
        print(f"\n  Overall Character Accuracy : {overall_acc:.1%}")
        print(f"  Total karakter             : {len(all_true)}")
        print(f"  Benar                      : {n_correct}")
        print(f"  Chance level (1/36)        : 2.8%")
        print(f"\n  Per subjek:")
        for r in results:
            print(f"    {r['subject']} ({r['session']}): "
                  f"{r['accuracy']:.1%} ({r['correct']}/{r['total']})")

        import pandas as pd
        df = pd.DataFrame(results)
        out_csv = os.path.join(BASE, "bci_decoding_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n  💾 Hasil: {out_csv}")
    else:
        print("  ❌ Tidak ada hasil!")


if __name__ == "__main__":
    main()