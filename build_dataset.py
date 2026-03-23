import os
import numpy as np
import pandas as pd

PREPROCESSED_DIR = "preprocessed_B_norm"
OUTPUT_DIR       = "."  


def get_files(cv_split):
    return sorted([
        f for f in os.listdir(PREPROCESSED_DIR)
        if f.endswith(".npz") and f"__{cv_split}__" in f
    ])


def build_split(cv_split):
    files = get_files(cv_split)
    if not files:
        print(f"❌ Tidak ada file {cv_split}!")
        return

    out_X = f"{cv_split}_X_B_norm.npy"
    out_y = f"{cv_split}_y_B_norm.npy"

    print(f"\n{'='*55}")
    print(f"  {cv_split}: {len(files)} file")
    print(f"{'='*55}")

    total = 0
    valid = []
    for fname in files:
        try:
            d = np.load(os.path.join(PREPROCESSED_DIR, fname), allow_pickle=True)
            total += len(d["labels"])
            valid.append(fname)
            d.close()
        except Exception as e:
            print(f"  ⚠️  {fname}: {e}")

    print(f"  📊 Total epoch : {total:,}")
    print(f"  📂 File valid  : {len(valid)}")

    tmp_X = f"/tmp/{cv_split}_X.dat"
    tmp_y = f"/tmp/{cv_split}_y.dat"
    X_mm  = np.memmap(tmp_X, dtype=np.float32, mode="w+", shape=(total, 32, 128))
    y_mm  = np.memmap(tmp_y, dtype=np.int8,    mode="w+", shape=(total,))

    cursor   = 0
    all_meta = []
    for i, fname in enumerate(valid):
        d      = np.load(os.path.join(PREPROCESSED_DIR, fname), allow_pickle=True)
        epochs = d["epochs"]
        labels = d["labels"]
        n      = len(labels)
        X_mm[cursor:cursor+n] = epochs
        y_mm[cursor:cursor+n] = labels
        cursor += n

        parts = fname.replace(".npz","").split("__")
        for _ in range(n):
            all_meta.append({
                "subject":   parts[0],
                "session":   parts[1],
                "cv_split":  parts[2],
            })

        del epochs, labels
        d.close()

        if (i+1) % 50 == 0:
            print(f"  ... {i+1}/{len(valid)} ({cursor:,} epoch)")

    X_mm.flush()
    y_mm.flush()

    print(f"  💾 Menyimpan {out_X}...")
    np.save(out_X, np.array(X_mm))
    np.save(out_y, np.array(y_mm))

    del X_mm, y_mm
    os.remove(tmp_X)
    os.remove(tmp_y)

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(f"{cv_split}_meta_B.csv", index=False)

    size = os.path.getsize(out_X) / 1024 / 1024 / 1024
    y_check = np.load(out_y)
    print(f"\n  ✅ {cv_split} selesai!")
    print(f"  📊 Epoch    : {len(y_check):,}")
    print(f"  🎯 Target   : {int((y_check==1).sum()):,}")
    print(f"  ❌ NonTarget: {int((y_check==0).sum()):,}")
    print(f"  👥 Subjek   : {meta_df['subject'].nunique()}")
    print(f"  💾 Ukuran   : {size:.2f} GB → {out_X}")


def main():
    print("="*55)
    print("  BUILD DATASET STUDY B — Cross-Session")
    print("="*55)

    build_split("Train")
    build_split("Test")

    print("\n" + "="*55)
    print("  SELESAI!")
    print("  File output:")
    print("  → Train_X_B.npy + Train_y_B.npy")
    print("  → Test_X_B.npy  + Test_y_B.npy")
    print("="*55)


if __name__ == "__main__":
    main()