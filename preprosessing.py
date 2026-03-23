import os
import numpy as np
import mne
import pandas as pd

mne.set_log_level("WARNING")

BASE_PATH  = "/Users/istiqomah/Dev/Kuliah smt4/IDSC/bigp3bci-an-open-diverse-and-machine-learning-ready-p300-based-brain-computer-interface-dataset-1.0.0"
INDEX_CSV  = "file_index_B.csv"
OUTPUT_DIR = "preprocessed_B"

SFREQ_NEW       = 128
L_FREQ          = 0.5
H_FREQ          = 40.0
TMIN            = -0.2
TMAX            = 0.8
NONTARGET_RATIO = 2
RANDOM_SEED     = 42

EEG_CHANNELS = [
    "EEG_F3",  "EEG_Fz",  "EEG_F4",  "EEG_T7",  "EEG_C3",  "EEG_Cz",
    "EEG_C4",  "EEG_T8",  "EEG_CP3", "EEG_CP4", "EEG_P3",  "EEG_Pz",
    "EEG_P4",  "EEG_PO7", "EEG_PO8", "EEG_Oz",  "EEG_FP1", "EEG_FP2",
    "EEG_F7",  "EEG_F8",  "EEG_FC5", "EEG_FC1", "EEG_FC2", "EEG_FC6",
    "EEG_CPz", "EEG_P7",  "EEG_P5",  "EEG_PO3", "EEG_POz", "EEG_PO4",
    "EEG_O1",  "EEG_O2"
]

EEG_MIN = [
    "EEG_F3", "EEG_Fz", "EEG_F4", "EEG_T7", "EEG_C3", "EEG_Cz",
    "EEG_C4", "EEG_T8", "EEG_CP3", "EEG_CP4", "EEG_P3", "EEG_Pz",
    "EEG_P4", "EEG_PO7", "EEG_PO8", "EEG_Oz"
]


def preprocess_one_file(filepath, rng):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    missing = [ch for ch in EEG_MIN if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Channel kurang: {missing}")

    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available)
    raw.set_channel_types({ch: "eeg" for ch in available})
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.resample(SFREQ_NEW, verbose=False)

    raw_ev     = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    stim_begin = raw_ev.get_data(picks=["StimulusBegin"])[0]
    stim_type  = raw_ev.get_data(picks=["StimulusType"])[0]
    sfreq_orig = raw_ev.info["sfreq"]
    del raw_ev

    onset_orig = np.where(stim_begin > 0)[0]
    if len(onset_orig) == 0:
        return None, None

    onset_new = (onset_orig / sfreq_orig * SFREQ_NEW).astype(int)
    labels    = stim_type[onset_orig].astype(int)

    tgt_idx   = np.where(labels == 1)[0]
    nt_idx    = np.where(labels == 0)[0]
    chosen_nt = rng.choice(nt_idx,
                           size=min(len(nt_idx), len(tgt_idx)*NONTARGET_RATIO),
                           replace=False)
    chosen_nt.sort()
    chosen = np.sort(np.concatenate([tgt_idx, chosen_nt]))

    eeg_data   = raw.get_data()
    n_baseline = int(abs(TMIN) * SFREQ_NEW)
    n_after    = int(TMAX * SFREQ_NEW) + 1

    epochs_list, labels_list = [], []
    for idx in chosen:
        onset = onset_new[idx]
        start, end = onset - n_baseline, onset + n_after
        if start < 0 or end > eeg_data.shape[1]:
            continue
        epoch = eeg_data[:, start:end].copy().astype(np.float32)
        epoch -= epoch[:, :n_baseline].mean(axis=1, keepdims=True)
        if epoch.shape[0] < 32:
            pad   = np.zeros((32 - epoch.shape[0], epoch.shape[1]), dtype=np.float32)
            epoch = np.vstack([epoch, pad])
        epochs_list.append(epoch)
        labels_list.append(labels[idx])

    del raw, eeg_data
    if not epochs_list:
        return None, None

    return (np.array(epochs_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int8))


def main():
    print("=" * 55)
    print("  PREPROCESSING STUDY B")
    print("=" * 55)

    df  = pd.read_csv(INDEX_CSV)
    df  = df[df["cv_split"].isin(["Train", "Test"])].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"📂 Total file: {len(df)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = error = 0
    for i, row in df.iterrows():
        stem     = os.path.splitext(row["filename"])[0]
        out_name = (f"{row['subject']}__{row['session']}__"
                    f"{row['cv_split']}__{row['orig_split']}__{stem}.npz")
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            success += 1
            continue

        print(f"  [{i+1}/{len(df)}] {row['filename']} ...", end=" ", flush=True)
        try:
            epochs, labels = preprocess_one_file(row["filepath"], rng)
            if epochs is None:
                print("⚠️  skip")
                error += 1
                continue
            np.savez_compressed(out_path, epochs=epochs, labels=labels)
            size = os.path.getsize(out_path) / 1024 / 1024
            print(f"✅ {epochs.shape} | {size:.1f}MB")
            success += 1
        except Exception as e:
            print(f"❌ {e}")
            error += 1

    print(f"\n{'='*55}")
    print(f"  SELESAI! ✅ {success} | ❌ {error}")
    total = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
                for f in os.listdir(OUTPUT_DIR) if f.endswith(".npz"))
    print(f"  💾 Total: {total:.0f} MB ({total/1024:.1f} GB)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()