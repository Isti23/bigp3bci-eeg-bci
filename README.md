# P300-Based BCI Decoding with EEG-Conformer
### IDSC 2026 — Mathematics for Hope in Healthcare

A cross-session P300 BCI speller decoding pipeline using EEG-Conformer on the bigP3BCI dataset (StudyB). This project addresses three core BCI challenges: P300 detection, BCI character decoding, and cross-session generalization.

---

## Results

| Task | Metric | Result |
|------|--------|--------|
| P300 Detection | AUC-ROC | **0.7631** |
| P300 Detection | Accuracy | **70.42%** |
| BCI Decoding | Character Accuracy | **57.7%** |
| Baseline (random) | Character Accuracy | 2.8% (1/36) |

Cross-session split: **Train = SE001**, **Test = last session** (per subject)

---

## Dataset

**bigP3BCI** — An Open, Diverse and Machine Learning Ready P300-based BCI Dataset  
Source: [PhysioNet](https://doi.org/10.13026/0byy-ry86)  
Study used: **StudyB** (13 subjects, multi-session, 6×6 checkerboard paradigm)

> ⚠️ Dataset is not included in this repository. Download from PhysioNet and place in `bigP3BCI-data/StudyB/`.

---

## Pipeline

```
StudyB (.edf files)
    │
    ├── load_data_B.py          → scan files, create file_index_B.csv
    ├── preprocessing_B.py      → filter, epoch, save .npz (raw Volt)
    ├── preprocessing_B_norm.py → filter, epoch, Z-score per subject, save .npz
    ├── build_dataset_B.py      → merge .npz → Train/Test .npy
    ├── train_model_B.py        → train EEG-Conformer (Google Colab, GPU)
    └── bci_decoding.py         → P300 detection + character decoding
```

---

## Cross-Session Split

| Split | Sessions | Purpose |
|-------|----------|---------|
| Train | SE001 | Model training |
| Middle | SE002–SE(n-1) | Not used |
| Test | Last session | Evaluation |

Only subjects with ≥2 sessions are included (13 out of 19 subjects).

---

## Model Architecture

**EEG-Conformer** — CNN + Transformer hybrid for EEG classification

- PatchEmbedding: depthwise Conv2D → BatchNorm → AvgPool
- ConformerBlock × 2: Multi-head Self-Attention + Feed-Forward
- Classifier: AdaptiveAvgPool → Linear → output
- Parameters: **41,025**
- Training: 60 epochs, AdamW, CosineAnnealingLR, BCEWithLogitsLoss + pos_weight

---

## Preprocessing

- Bandpass filter: **0.5–40 Hz**
- Resampling: **256 Hz → 128 Hz**
- Epoch window: **-200ms to +800ms** from stimulus onset
- Baseline correction: mean of pre-stimulus period
- NonTarget subsampling: **1:2** (Target:NonTarget)
- Normalization: **Z-score per subject** (mean & std from SE001 Train)

---

## BCI Decoding

Uses the Checkerboard (CB) paradigm where each flash activates 4 out of 36 characters simultaneously.

```
For each character trial:
  1. Predict P300 probability for each flash epoch
  2. Accumulate probability scores per character
  3. Select character with highest accumulated score
```

---

## Reproducibility

```bash
# Step 1 — Scan dataset (Mac/local)
python3 load_data_B.py

# Step 2 — Preprocessing (Mac/local)
caffeinate -i python3 preprocessing_B.py
caffeinate -i python3 preprocessing_B_norm.py

# Step 3 — Build dataset (Mac/local)
python3 build_dataset_B.py

# Step 4 — Upload Train_X_B_norm.npy, Train_y_B_norm.npy,
#           Test_X_B_norm.npy, Test_y_B_norm.npy to Google Drive

# Step 5 — Train model (Google Colab, GPU T4)
# Run train_model_B.py in Colab

# Step 6 — BCI Decoding (Mac/local)
python3 bci_decoding.py
```

---

## Requirements

```
pip install -r requirements.txt
```

See `requirements.txt` for full dependencies.

---

## Citation

```
Mainsah, B., Fleeting, C., Balmat, T., Sellers, E., & Collins, L. (2025).
bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based Brain-Computer
Interface Dataset (version 1.0.0). PhysioNet.
https://doi.org/10.13026/0byy-ry86
```

---

## Ethical Considerations

- Dataset is fully anonymized (no personal identifiers)
- All original studies were IRB-approved
- Model interpretability: EEG-Conformer attention weights can be visualized to identify P300 temporal patterns
- Intended use: assistive communication technology for individuals with ALS and severe motor disabilities
