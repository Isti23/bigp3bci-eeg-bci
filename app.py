import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import mne
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import os

mne.set_log_level("WARNING")

st.set_page_config(
    page_title="P300 BCI Decoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; color: #e0e6f0; }

    .metric-card {
        background: linear-gradient(135deg, #111827, #1a2332);
        border: 1px solid #2d3a4f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }
    .result-box {
        background: linear-gradient(135deg, #0f2027, #1a3a2a);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .result-char {
        font-family: 'Space Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: #22c55e;
    }
    .result-wrong {
        border-color: #ef4444;
        background: linear-gradient(135deg, #0f2027, #3a1a1a);
    }
    .result-char-wrong {
        color: #ef4444;
    }
    .info-box {
        background: #111827;
        border-left: 3px solid #38bdf8;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        transform: translateY(-1px);
    }
    div[data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #2d3a4f;
        border-radius: 12px;
        padding: 16px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 6px;
        margin: 16px 0;
    }
    .grid-cell {
        background: #1a2332;
        border: 1px solid #2d3a4f;
        border-radius: 6px;
        padding: 10px 4px;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: #94a3b8;
        transition: all 0.3s;
    }
    .grid-cell-predicted {
        background: linear-gradient(135deg, #0f2027, #1a3a2a);
        border-color: #22c55e;
        color: #22c55e;
        font-weight: 700;
    }
    .grid-cell-true {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-color: #38bdf8;
        color: #38bdf8;
        font-weight: 700;
    }
    .grid-cell-both {
        background: linear-gradient(135deg, #0f2027, #1a3a2a);
        border-color: #22c55e;
        color: #22c55e;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(34,197,94,0.4);
    }
    .sidebar-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

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


# ── Constants ────────────────────────────────────────────────
EEG_CHANNELS = [
    "EEG_F3","EEG_Fz","EEG_F4","EEG_T7","EEG_C3","EEG_Cz",
    "EEG_C4","EEG_T8","EEG_CP3","EEG_CP4","EEG_P3","EEG_Pz",
    "EEG_P4","EEG_PO7","EEG_PO8","EEG_Oz","EEG_FP1","EEG_FP2",
    "EEG_F7","EEG_F8","EEG_FC5","EEG_FC1","EEG_FC2","EEG_FC6",
    "EEG_CPz","EEG_P7","EEG_P5","EEG_PO3","EEG_POz","EEG_PO4",
    "EEG_O1","EEG_O2"
]
SFREQ_NEW = 128
L_FREQ, H_FREQ = 0.5, 40.0
TMIN, TMAX = -0.2, 0.8
N_COLS = 6


# ── Helper functions ─────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    model = EEGConformer()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    return model

@st.cache_resource
def load_subject_stats(prep_dir, subject):
    files = [f for f in os.listdir(prep_dir)
             if f.startswith(subject) and "__SE001__Train__" in f]
    if not files:
        return None, None
    all_epochs = []
    for fname in files:
        d = np.load(os.path.join(prep_dir, fname))
        all_epochs.append(d["epochs"])
        d.close()
    X    = np.concatenate(all_epochs, axis=0)
    mean = X.mean(axis=(0, 2), keepdims=True)
    std  = X.std(axis=(0, 2),  keepdims=True) + 1e-8
    return mean, std

def build_charidx_to_target(char_channels, n_cols=6):
    mapping = {}
    for idx, ch in enumerate(char_channels):
        parts = ch.split('_')
        row, col = int(parts[-2]), int(parts[-1])
        mapping[idx] = (row - 1) * n_cols + col
    return mapping

def target_to_char(char_channels, target_num):
    for ch in char_channels:
        parts = ch.split('_')
        row, col = int(parts[-2]), int(parts[-1])
        if (row - 1) * N_COLS + col == target_num:
            return parts[0]
    return "?"

def decode_edf(filepath, model, subject_mean, subject_std):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    char_channels = sorted([ch for ch in raw.ch_names
                             if '_' in ch and len(ch.split('_')) == 3])
    if len(char_channels) != 36:
        return None, None, None, None

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
    eeg_data = raw.get_data()

    onset_orig = np.where(stim_begin > 0)[0]
    onset_new  = (onset_orig / sfreq_orig * SFREQ_NEW).astype(int)

    n_baseline = int(abs(TMIN) * SFREQ_NEW)
    n_after    = int(TMAX * SFREQ_NEW) + 1

    epochs_list, active_targets_list, targets_list = [], [], []
    erp_target, erp_nontarget = [], []
    stim_type_all = raw.get_data() if False else None

    raw2 = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    stim_type_orig = raw2.get_data(picks=["StimulusType"])[0]

    for o_orig, o_new in zip(onset_orig, onset_new):
        start, end = o_new - n_baseline, o_new + n_after
        if start < 0 or end > eeg_data.shape[1]:
            continue
        epoch = eeg_data[:, start:end].copy().astype(np.float32)
        epoch -= epoch[:, :n_baseline].mean(axis=1, keepdims=True)
        if epoch.shape[0] < 32:
            pad   = np.zeros((32 - epoch.shape[0], epoch.shape[1]), dtype=np.float32)
            epoch = np.vstack([epoch, pad])
        epoch = epoch[:32, :128]
        if epoch.shape[1] < 128:
            epoch = np.pad(epoch, ((0, 0), (0, 128 - epoch.shape[1])))

        if stim_type_orig[o_orig] == 1:
            erp_target.append(epoch[15])
        else:
            erp_nontarget.append(epoch[15])

        if subject_mean is not None:
            epoch_norm = (epoch - subject_mean[0]) / subject_std[0]
        else:
            epoch_norm = epoch

        active_tgt_nums = [charidx_to_target[i]
                           for i in np.where(char_data[:, o_orig] > 0)[0]
                           if i in charidx_to_target]
        epochs_list.append(epoch_norm)
        active_targets_list.append(active_tgt_nums)
        targets_list.append(int(current_tgt[o_orig]))

    if not epochs_list:
        return None, None, None, None

    epochs_arr = np.stack(epochs_list, axis=0)

    probs = []
    with torch.no_grad():
        for i in range(0, len(epochs_arr), 256):
            batch = torch.tensor(epochs_arr[i:i+256],
                                 dtype=torch.float32).unsqueeze(1)
            p = torch.sigmoid(model(batch)).numpy()
            probs.extend(p.tolist())
    probs = np.array(probs)
    
    all_tgt_nums = sorted(set(charidx_to_target.values()))
    tgt_to_idx   = {t: i for i, t in enumerate(all_tgt_nums)}

    trials = []
    current_trial = None
    for p, active_tgts, tgt in zip(probs, active_targets_list, targets_list):
        if current_trial is None or current_trial["target"] != tgt:
            current_trial = {"scores": np.zeros(len(all_tgt_nums)), "target": tgt}
            trials.append(current_trial)
        for tgt_num in active_tgts:
            if tgt_num in tgt_to_idx:
                current_trial["scores"][tgt_to_idx[tgt_num]] += p

    results = []
    for trial in trials:
        best_idx      = int(np.argmax(trial["scores"]))
        predicted_tgt = all_tgt_nums[best_idx]
        true_char     = target_to_char(char_channels, trial["target"])
        pred_char     = target_to_char(char_channels, predicted_tgt)
        results.append({
            "true_tgt":  trial["target"],
            "pred_tgt":  predicted_tgt,
            "true_char": true_char,
            "pred_char": pred_char,
            "correct":   trial["target"] == predicted_tgt,
            "scores":    trial["scores"],
            "all_tgts":  all_tgt_nums,
        })

    erp_data = {
        "target":    np.array(erp_target).mean(axis=0) if erp_target else None,
        "nontarget": np.array(erp_nontarget).mean(axis=0) if erp_nontarget else None,
    }

    return results, char_channels, erp_data, all_tgt_nums


def plot_erp(erp_data):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#111827')
    t = np.linspace(TMIN, TMAX, 128) * 1000

    if erp_data["target"] is not None:
        ax.plot(t, erp_data["target"] * 1e6, color="#22c55e",
                linewidth=2, label="Target (P300)")
    if erp_data["nontarget"] is not None:
        ax.plot(t, erp_data["nontarget"] * 1e6, color="#94a3b8",
                linewidth=1.5, alpha=0.7, label="Non-Target")

    ax.axvline(x=300, color="#38bdf8", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0,   color="#ffffff", linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=0,   color="#ffffff", alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Time (ms)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Amplitude (µV)", color="#94a3b8", fontsize=10)
    ax.set_title("ERP Waveform — Oz Channel", color="#e0e6f0",
                 fontsize=11, fontfamily="monospace")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3a4f")
    ax.legend(facecolor="#111827", edgecolor="#2d3a4f",
              labelcolor="#e0e6f0", fontsize=9)
    ax.annotate("P300", xy=(300, 0), xytext=(320, 0),
                color="#38bdf8", fontsize=8, alpha=0.7)
    plt.tight_layout()
    return fig


def render_grid(char_channels, pred_tgt, true_tgt):
    grid = {}
    for ch in char_channels:
        parts = ch.split('_')
        char_name = parts[0]
        row, col  = int(parts[-2]), int(parts[-1])
        tgt_num   = (row - 1) * N_COLS + col
        grid[(row, col)] = (char_name, tgt_num)

    html = '<div class="grid-container">'
    for row in range(1, 7):
        for col in range(1, 7):
            if (row, col) in grid:
                char_name, tgt_num = grid[(row, col)]
                is_pred = tgt_num == pred_tgt
                is_true = tgt_num == true_tgt
                if is_pred and is_true:
                    css = "grid-cell grid-cell-both"
                elif is_pred:
                    css = "grid-cell grid-cell-predicted"
                elif is_true:
                    css = "grid-cell grid-cell-true"
                else:
                    css = "grid-cell"
                html += f'<div class="{css}">{char_name}</div>'
    html += '</div>'

    legend = """
    <div style="display:flex; gap:16px; margin-top:8px; font-size:0.75rem; color:#94a3b8;">
        <span><span style="color:#22c55e">■</span> Predicted</span>
        <span><span style="color:#38bdf8">■</span> True target</span>
        <span><span style="color:#22c55e">■ (glow)</span> Correct!</span>
    </div>
    """
    return html + legend


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙ Configuration</p>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model path (.pth)",
        value="best_conformer_B.pth",
        help="Path ke file model EEG-Conformer"
    )

    prep_dir = st.text_input(
        "Preprocessed dir",
        value="preprocessed_B",
        help="Folder preprocessed_B untuk hitung stats normalisasi"
    )

    subject_id = st.text_input(
        "Subject ID",
        value="B_01",
        help="Contoh: B_01, B_02, ..., B_20"
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-title">📊 About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Dataset</b>: bigP3BCI StudyB<br>
    <b>Model</b>: EEG-Conformer<br>
    <b>Grid</b>: 6×6 (36 chars)<br>
    <b>Paradigm</b>: Checkerboard (CB)<br>
    <b>P300 Detection AUC</b>: 0.7631<br>
    <b>Char Accuracy</b>: 57.7%
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">📚 Citation</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-size:0.75rem;">
    Mainsah et al. (2025). bigP3BCI.
    PhysioNet. doi:10.13026/0byy-ry86
    </div>
    """, unsafe_allow_html=True)


# ── Main UI ──────────────────────────────────────────────────
st.markdown("# 🧠 P300 BCI Decoder")
st.markdown("**IDSC 2026** — Mathematics for Hope in Healthcare | EEG-Conformer × Cross-Session Generalization")
st.markdown("---")

st.markdown("### 📂 Upload EDF File")
st.markdown('<div class="info-box">Upload file <b>.edf</b> dari StudyB Test session. Model akan memprediksi karakter yang dituju subjek berdasarkan sinyal EEG.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if uploaded_file:
    col1, col2, col3 = st.columns(3)

    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan: `{model_path}`")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model(model_path)

    mean, std = None, None
    if os.path.exists(prep_dir):
        mean, std = load_subject_stats(prep_dir, subject_id)

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Decoding EEG..."):
        results, char_channels, erp_data, all_tgt_nums = decode_edf(
            tmp_path, model, mean, std)
    os.unlink(tmp_path)

    if results is None:
        st.error("❌ Gagal memproses file. Pastikan format EDF benar.")
        st.stop()

    # ── Metrics ──────────────────────────────────────────────
    st.markdown("### 📊 Results")
    n_correct = sum(r["correct"] for r in results)
    n_total   = len(results)
    acc       = n_correct / n_total if n_total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_correct}/{n_total}</div>
            <div class="metric-label">Characters Correct</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc*100:.0f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">2.8%</div>
            <div class="metric-label">Chance Level (1/36)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔤 Character Predictions")
    for i, r in enumerate(results):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            correct_emoji = "✅" if r["correct"] else "❌"
            box_class = "result-box" if r["correct"] else "result-box result-wrong"
            char_class = "result-char" if r["correct"] else "result-char result-char-wrong"
            st.markdown(f"""
            <div class="{box_class}">
                <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">
                    Trial {i+1} {correct_emoji}
                </div>
                <div class="{char_class}">{r['pred_char']}</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                    Predicted → <b style="color:#e0e6f0">{r['pred_char']}</b>
                    &nbsp;|&nbsp; True → <b style="color:#38bdf8">{r['true_char']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**6×6 Speller Grid:**")
            grid_html = render_grid(char_channels, r["pred_tgt"], r["true_tgt"])
            st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown("")

    # ── ERP Plot ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 ERP Waveform (Oz Channel)")
    st.markdown('<div class="info-box">Average EEG response to Target vs Non-Target stimuli. The P300 component (~300ms) is visible in the Target waveform — this is what the model detects.</div>', unsafe_allow_html=True)

    if erp_data["target"] is not None:
        fig = plot_erp(erp_data)
        st.pyplot(fig)
    else:
        st.info("ERP data not available.")

else:

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.763</div>
            <div class="metric-label">P300 Detection AUC</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">57.7%</div>
            <div class="metric-label">Character Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">13</div>
            <div class="metric-label">Subjects (Cross-Session)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>How it works:</b><br>
    1. Upload a <code>.edf</code> file from bigP3BCI StudyB Test session<br>
    2. The EEG-Conformer model detects P300 responses in each flash epoch<br>
    3. Probabilities are accumulated per character → highest score = predicted character<br>
    4. Results shown with speller grid visualization and ERP waveform
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>EEG-Conformer</b><br>
        • PatchEmbedding (depthwise CNN)<br>
        • 2× ConformerBlock (Attention + FFN)<br>
        • AdaptiveAvgPool + Linear classifier<br>
        • 41,025 parameters
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <b>Cross-Session Split</b><br>
        • Train: SE001 (first session)<br>
        • Test: Last session per subject<br>
        • 13 subjects with ≥2 sessions<br>
        • Normalization: Z-score per subject
        </div>
        """, unsafe_allow_html=True)