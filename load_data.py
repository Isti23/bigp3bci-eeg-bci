import os
import pandas as pd


BASE_PATH  = "/Users/istiqomah/Dev/Kuliah smt4/IDSC/bigp3bci-an-open-diverse-and-machine-learning-ready-p300-based-brain-computer-interface-dataset-1.0.0"
STUDY_DIR  = os.path.join(BASE_PATH, "bigP3BCI-data", "StudyB")
OUTPUT_CSV = "file_index_B.csv"

VALID_SUBJECTS = [
    "B_01", "B_02", "B_03", "B_04", "B_06",
    "B_10", "B_11", "B_12", "B_13", "B_14",
    "B_17", "B_18", "B_20"
]

def get_last_session(subject_path):
    sessions = sorted([
        s for s in os.listdir(subject_path)
        if os.path.isdir(os.path.join(subject_path, s))
        and s.startswith("SE")
    ])
    return sessions[-1] if sessions else None

def scan_study_b():
    records = []

    for subject_name in sorted(os.listdir(STUDY_DIR)):
        if subject_name not in VALID_SUBJECTS:
            continue

        subject_path = os.path.join(STUDY_DIR, subject_name)
        if not os.path.isdir(subject_path):
            continue

        last_session = get_last_session(subject_path)
        sessions     = sorted([
            s for s in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, s))
        ])

        print(f"  {subject_name}: {sessions} → last={last_session}")

        for session_name in sessions:
            session_path = os.path.join(subject_path, session_name)

            if session_name == "SE001":
                cv_split = "Train"
            elif session_name == last_session:
                cv_split = "Test"
            else:
                cv_split = "Middle" 

            for split_name in ["Train", "Test"]:
                split_path = os.path.join(session_path, split_name)
                if not os.path.isdir(split_path):
                    continue

                cb_path = os.path.join(split_path, "CB")
                if not os.path.isdir(cb_path):
                    continue

                for filename in sorted(os.listdir(cb_path)):
                    if not filename.endswith(".edf"):
                        continue

                    records.append({
                        "study":      "StudyB",
                        "subject":    subject_name,
                        "session":    session_name,
                        "cv_split":   cv_split,    
                        "orig_split": split_name,  
                        "paradigm":   "CB",
                        "filename":   filename,
                        "filepath":   os.path.join(cb_path, filename),
                    })

    return records

def main():
    print("=" * 55)
    print("  SCAN STUDY B — Cross-Session Split")
    print("=" * 55)

    if not os.path.exists(STUDY_DIR):
        print(f"❌ Folder tidak ditemukan: {STUDY_DIR}")
        return

    print(f"\n📂 Subjek valid (≥2 session): {VALID_SUBJECTS}\n")
    records = scan_study_b()

    df = pd.DataFrame(records)

    print(f"\n{'='*55}")
    print(f"  RINGKASAN")
    print(f"{'='*55}")
    print(f"✅ Total file .edf : {len(df)}")
    print(f"   Subjek          : {df['subject'].nunique()}")
    print(f"\n📊 File per subjek & cv_split:")
    print(df.groupby(["subject", "cv_split"]).size().to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Disimpan ke: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()