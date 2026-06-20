# -*- coding: utf-8 -*-
"""
================================================================
  TASK 2 -- Emotion Recognition from Speech
  CodeAlpha Machine Learning Internship
================================================================
  Pipeline:
    1. Synthetic audio generation  (RAVDESS-style label scheme)
    2. Feature extraction: MFCCs, Delta-MFCCs, Chroma, ZCR, RMS
    3. Four classifiers: SVM | MLP | Random Forest | Gradient Boosting
    4. Evaluation: Accuracy, F1, AUC-ROC, Confusion Matrix
    5. Full visual report saved as PNG
================================================================
  NOTE: Rewritten with scikit-learn models (Python 3.13 compatible).
        TensorFlow / Keras is NOT required.
================================================================
"""

import sys, io, os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches
import seaborn as sns

import librosa
import librosa.display

from pathlib import Path

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm              import SVC
from sklearn.neural_network   import MLPClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.pipeline         import Pipeline

np.random.seed(42)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================
# GLOBAL STYLE
# ==============================================================
BG   = "#0F172A"
CARD = "#1E293B"
TEXT = "#F1F5F9"
GRID = "#334155"
ACC1 = "#6366F1"   # SVM
ACC2 = "#10B981"   # MLP
ACC3 = "#F59E0B"   # Random Forest
ACC4 = "#EF4444"   # Gradient Boosting
MODEL_COLORS = [ACC1, ACC2, ACC3, ACC4]

EMO_COLORS = ["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6","#EC4899","#14B8A6"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  GRID,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ==============================================================
# 1.  SYNTHETIC AUDIO GENERATION (RAVDESS-inspired)
#     Each emotion -> characteristic frequency + modulation
# ==============================================================
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SR                   = 22050   # sample rate
DURATION             = 2.5     # seconds per clip
SAMPLES_PER_EMOTION  = 120

EMOTION_PARAMS = {
    "neutral":   dict(freq=200, mod=0.2, noise=0.03, pitch_var=0.5),
    "happy":     dict(freq=320, mod=0.7, noise=0.04, pitch_var=1.5),
    "sad":       dict(freq=130, mod=0.1, noise=0.02, pitch_var=0.3),
    "angry":     dict(freq=280, mod=0.9, noise=0.08, pitch_var=2.0),
    "fearful":   dict(freq=240, mod=0.6, noise=0.07, pitch_var=1.8),
    "disgust":   dict(freq=160, mod=0.4, noise=0.05, pitch_var=0.8),
    "surprised": dict(freq=350, mod=0.8, noise=0.06, pitch_var=2.2),
}


def synthesize_emotion(emotion: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Generate a synthetic audio signal characteristic of the given emotion."""
    p = EMOTION_PARAMS[emotion]
    t = np.linspace(0, duration, int(sr * duration))
    # Base tone + harmonic overtones
    sig  = np.sin(2 * np.pi * p["freq"] * t)
    sig += 0.5 * np.sin(2 * np.pi * p["freq"] * 2 * t)
    sig += 0.25 * np.sin(2 * np.pi * p["freq"] * 3 * t)
    # Amplitude modulation
    mod  = 1 + p["mod"] * np.sin(2 * np.pi * 4 * t)
    # Pitch variation (vibrato)
    pitch_shift = p["pitch_var"] * np.sin(2 * np.pi * 5 * t)
    sig_var = np.sin(2 * np.pi * (p["freq"] + pitch_shift) * t)
    sig = 0.6 * sig * mod + 0.4 * sig_var
    # Gaussian noise
    sig += p["noise"] * np.random.randn(len(t))
    # Normalize
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig.astype(np.float32)


print("=" * 60)
print("  EMOTION RECOGNITION FROM SPEECH")
print("  CodeAlpha ML Internship -- Task 2")
print("=" * 60)
print("\n[1/4] Generating synthetic audio clips ...")
audio_data, labels = [], []
for emotion in EMOTIONS:
    for _ in range(SAMPLES_PER_EMOTION):
        sig = synthesize_emotion(emotion)
        # Small random tempo variation for augmentation
        stretch = np.random.uniform(0.9, 1.1)
        sig = librosa.effects.time_stretch(sig, rate=stretch)
        sig = librosa.util.fix_length(sig, size=int(SR * DURATION))
        audio_data.append(sig)
        labels.append(emotion)

print(f"   [OK] {len(audio_data)} clips | {len(EMOTIONS)} emotions")

# ==============================================================
# 2.  FEATURE EXTRACTION
#     Flat vector: MFCCs + deltas + Chroma + ZCR + RMS
# ==============================================================
N_MFCC = 40
N_MELS = 64
HOP    = 512
N_FFT  = 2048


def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a fixed-size flat feature vector from a raw audio signal."""
    mfcc        = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                       n_fft=N_FFT, hop_length=HOP)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma      = librosa.feature.chroma_stft(y=signal, sr=sr,
                                               n_fft=N_FFT, hop_length=HOP)
    zcr         = librosa.feature.zero_crossing_rate(signal, hop_length=HOP)
    rms         = librosa.feature.rms(y=signal, hop_length=HOP)

    feats = np.concatenate([
        np.mean(mfcc,        axis=1), np.std(mfcc,        axis=1),
        np.mean(mfcc_delta,  axis=1), np.std(mfcc_delta,  axis=1),
        np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
        np.mean(chroma,      axis=1), np.std(chroma,      axis=1),
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)],
    ])
    return feats


print("[2/4] Extracting audio features ...")
X = np.array([extract_features(sig) for sig in audio_data], dtype=np.float32)

le    = LabelEncoder()
y_enc = le.fit_transform(labels)
n_cls = len(EMOTIONS)
print(f"   [OK] Feature matrix: {X.shape} | Classes: {n_cls}")

# ==============================================================
# 3.  TRAIN / TEST SPLIT + SCALING
# ==============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# ==============================================================
# 4.  MODELS
# ==============================================================
MODELS = {
    "SVM":               SVC(kernel="rbf", C=10, gamma="scale",
                             probability=True, random_state=42),
    "MLP":               MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                       max_iter=500, early_stopping=True,
                                       validation_fraction=0.15,
                                       random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=300,
                                                max_depth=None,
                                                random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                     max_depth=5,
                                                     learning_rate=0.1,
                                                     random_state=42),
}

# Models that need scaled input
NEEDS_SCALE = {"SVM", "MLP"}

print("[3/4] Training & evaluating models (5-fold CV) ...")
print(f"\n  {'Model':<22} {'Acc':>6} {'F1':>6} {'AUC':>6}  {'CV-Acc':>8}")
print(f"  {'-'*58}")

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
history = {}          # for learning curves on MLP

for name, clf in MODELS.items():
    Xtr = X_tr_sc if name in NEEDS_SCALE else X_train
    Xte = X_te_sc if name in NEEDS_SCALE else X_test

    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    y_bin = label_binarize(y_test, classes=range(n_cls))
    auc_s = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")

    cv_acc = cross_val_score(clf, Xtr, y_train, cv=cv,
                              scoring="accuracy", n_jobs=-1).mean()

    results[name] = dict(acc=acc, f1=f1, auc=auc_s, cv_acc=cv_acc,
                         prob=y_prob, pred=y_pred, clf=clf)

    print(f"  {name:<22} {acc:>6.4f} {f1:>6.4f} {auc_s:>6.4f}  {cv_acc:>8.4f}")

    # Capture MLP loss history for learning curve
    if name == "MLP":
        history["MLP_loss"]     = clf.loss_curve_
        history["MLP_val_loss"] = clf.validation_scores_

best_name = max(results, key=lambda k: results[k]["f1"])
print(f"\n  [BEST] {best_name}  "
      f"(Acc={results[best_name]['acc']:.4f}  "
      f"F1={results[best_name]['f1']:.4f}  "
      f"AUC={results[best_name]['auc']:.4f})")

# ==============================================================
# 5.  FEATURE IMPORTANCE  (Random Forest)
# ==============================================================
rf_clf    = results["Random Forest"]["clf"]
feat_imp  = rf_clf.feature_importances_
top_n     = 15
top_idx   = np.argsort(feat_imp)[-top_n:][::-1]

# ==============================================================
# 6.  VISUALISATION  (5-row, 3-column figure)
# ==============================================================
print("\n[4/4] Generating visual report ...")

fig = plt.figure(figsize=(22, 28), facecolor=BG)
fig.suptitle("Emotion Recognition from Speech  --  ML Pipeline",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99, family="monospace")

gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.54, wspace=0.38,
                       left=0.06, right=0.97, top=0.97, bottom=0.03)


def card_ax(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return ax


# ── ROW 0: MFCC plots for 3 emotions ──────────────────────────
for col, emo in enumerate(["happy", "angry", "sad"]):
    ax = card_ax(fig.add_subplot(gs[0, col]))
    sig  = synthesize_emotion(emo)
    mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=13, hop_length=HOP)
    librosa.display.specshow(mfcc, sr=SR, hop_length=HOP, x_axis="time",
                             ax=ax, cmap="magma")
    ax.set_title(f"MFCC  --  {emo.capitalize()}", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficient")

# ── ROW 1: Log-Mel Spectrograms for 3 emotions ────────────────
for col, emo in enumerate(["fearful", "disgust", "surprised"]):
    ax = card_ax(fig.add_subplot(gs[1, col]))
    sig     = synthesize_emotion(emo)
    mel     = librosa.feature.melspectrogram(y=sig, sr=SR, n_mels=N_MELS, hop_length=HOP)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_mel, sr=SR, hop_length=HOP,
                             x_axis="time", y_axis="mel",
                             ax=ax, cmap="viridis")
    ax.set_title(f"Mel-Spectrogram  --  {emo.capitalize()}", fontsize=11)

# ── ROW 2a: MLP loss curve + CV accuracy bars ─────────────────
ax_lc = card_ax(fig.add_subplot(gs[2, :2]))
if "MLP_loss" in history:
    ep = range(1, len(history["MLP_loss"]) + 1)
    ax_lc.plot(ep, history["MLP_loss"],     color=ACC2, lw=2,
               label="MLP Train Loss")
    # val scores are accuracy; invert for loss proxy
    val_loss_proxy = [1 - v for v in history["MLP_val_loss"]]
    ax_lc.plot(range(1, len(val_loss_proxy) + 1), val_loss_proxy,
               color=ACC2, lw=1.5, linestyle="--", alpha=0.75,
               label="MLP Val (1-acc)")
ax_lc.set_title("MLP Training Loss Curve", fontsize=12)
ax_lc.set_xlabel("Iteration")
ax_lc.set_ylabel("Loss")
ax_lc.legend(fontsize=9)
ax_lc.grid(color=GRID, alpha=0.4)
ax_lc.spines[["top", "right"]].set_visible(False)

# ── ROW 2b: Model metric comparison bar chart ─────────────────
ax_bar = card_ax(fig.add_subplot(gs[2, 2]))
model_names = list(results.keys())
metrics_k   = ["acc", "f1", "auc"]
xlabs       = ["Accuracy", "F1-Score", "ROC-AUC"]
x    = np.arange(3)
w    = 0.18
for i, (name, col) in enumerate(zip(model_names, MODEL_COLORS)):
    vals = [results[name][m] for m in metrics_k]
    bars = ax_bar.bar(x + i * w, vals, w, color=col, alpha=0.85, zorder=3,
                      edgecolor=BG, linewidth=0.5)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
ax_bar.set_xticks(x + w * 1.5)
ax_bar.set_xticklabels(xlabs, fontsize=9)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_title("Model Metrics Comparison", fontsize=11)
ax_bar.yaxis.grid(True, color=GRID, alpha=0.4, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.spines[["top", "right"]].set_visible(False)
handles = [mpatches.Patch(color=c, label=n)
           for n, c in zip(model_names, MODEL_COLORS)]
ax_bar.legend(handles=handles, fontsize=7, loc="upper left")

# ── ROW 3: Confusion matrices for all 4 models ────────────────
short_labels = [e[:4].capitalize() for e in EMOTIONS]
for col, (name, col_c) in enumerate(zip(list(results.keys())[:3], MODEL_COLORS[:3])):
    ax_cm = card_ax(fig.add_subplot(gs[3, col]))
    cm      = confusion_matrix(y_test, results[name]["pred"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax_cm, cbar=False, annot_kws={"size": 7})
    ax_cm.set_title(f"{name}\nConfusion Matrix", fontsize=10)
    ax_cm.set_xlabel("Predicted", fontsize=9)
    ax_cm.set_ylabel("Actual",    fontsize=9)
    ax_cm.tick_params(axis="x", rotation=30)

# ── ROW 4a: Per-emotion F1 for best model ─────────────────────
ax_f1 = card_ax(fig.add_subplot(gs[4, :2]))
report   = classification_report(y_test, results[best_name]["pred"],
                                  target_names=EMOTIONS, output_dict=True)
f1_per   = [report[e]["f1-score"] for e in EMOTIONS]
bars_f1  = ax_f1.bar(EMOTIONS, f1_per, color=EMO_COLORS, alpha=0.88, zorder=3,
                      edgecolor=BG, linewidth=0.5)
for b, v in zip(bars_f1, f1_per):
    ax_f1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
               f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)
ax_f1.set_ylim(0, 1.18)
ax_f1.set_title(f"Per-Emotion F1-Score  ({best_name} -- best model)", fontsize=11)
ax_f1.set_ylabel("F1-Score")
ax_f1.set_xlabel("Emotion")
ax_f1.yaxis.grid(True, color=GRID, alpha=0.4, zorder=0)
ax_f1.set_axisbelow(True)
ax_f1.spines[["top", "right"]].set_visible(False)
ax_f1.tick_params(axis="x", rotation=15)

# ── ROW 4b: ROC curves (best model, one-vs-rest) ──────────────
ax_roc = card_ax(fig.add_subplot(gs[4, 2]))
y_bin  = label_binarize(y_test, classes=range(n_cls))
best_prob = results[best_name]["prob"]
for i, (emo, ec) in enumerate(zip(EMOTIONS, EMO_COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], best_prob[:, i])
    ax_roc.plot(fpr, tpr, color=ec, lw=1.5,
                label=f"{emo[:5]} ({auc(fpr, tpr):.2f})")
ax_roc.plot([0, 1], [0, 1], "--", color=GRID, lw=1)
ax_roc.set_xlabel("FPR")
ax_roc.set_ylabel("TPR")
ax_roc.set_title(f"ROC Curves  ({best_name})", fontsize=11)
ax_roc.legend(fontsize=7)
ax_roc.grid(color=GRID, alpha=0.4)
ax_roc.spines[["top", "right"]].set_visible(False)

out_path = OUTPUT_DIR / "emotion_recognition_report.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  [SAVED] {out_path}")

# ── Final console summary ──────────────────────────────────────
print("\n" + "=" * 60)
print(f"  BEST MODEL : {best_name}")
print(f"  Accuracy   : {results[best_name]['acc']:.4f}")
print(f"  F1-Score   : {results[best_name]['f1']:.4f}")
print(f"  ROC-AUC    : {results[best_name]['auc']:.4f}")
print("=" * 60)
print(classification_report(y_test, results[best_name]["pred"],
                             target_names=EMOTIONS))