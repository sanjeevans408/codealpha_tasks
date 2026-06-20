"""
Emotion Recognition from Speech
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline:
  1. Synthetic audio signal generation (RAVDESS-style label scheme)
  2. Feature extraction: MFCCs, Chroma, Mel-Spectrogram, ZCR, RMS
  3. Three models: CNN (spectral), LSTM (sequential), Hybrid CNN-LSTM
  4. Evaluation: Accuracy, F1, Confusion Matrix, ROC-AUC
  5. Full visual report saved as PNG
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc
)

import sys
import traceback

python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 13:
    print(f"WARNING: Python {sys.version.split()[0]} is installed. TensorFlow 2.x is not officially supported on Python 3.13+.")
    print("Create a virtual environment with Python 3.11 or 3.12 and install dependencies there.\n")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.utils import to_categorical
except Exception as exc:
    print("ERROR: TensorFlow failed to import.\n")
    print(f"Python version: {sys.version.split()[0]}")
    print("Please ensure the active virtual environment has a compatible TensorFlow package installed.")
    print("Recommended command:")
    print("  pip install \"tensorflow-cpu>=2.12,<2.22\"")
    print("If pip download fails, download the wheel from https://pypi.org/project/tensorflow-cpu/ and install it manually.")
    print("Also verify the Microsoft Visual C++ Redistributable is installed on Windows.\n")
    traceback.print_exception(exc, exc, exc.__traceback__)
    sys.exit(1)

tf.random.set_seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════
# 1. SYNTHETIC AUDIO GENERATION (RAVDESS-inspired)
#    Each emotion → characteristic frequency + modulation
# ═══════════════════════════════════════════════════════
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SR = 22050        # sample rate
DURATION = 2.5    # seconds per clip
SAMPLES_PER_EMOTION = 120

EMOTION_PARAMS = {
    'neutral':   dict(freq=200, mod=0.2, noise=0.03, pitch_var=0.5),
    'happy':     dict(freq=320, mod=0.7, noise=0.04, pitch_var=1.5),
    'sad':       dict(freq=130, mod=0.1, noise=0.02, pitch_var=0.3),
    'angry':     dict(freq=280, mod=0.9, noise=0.08, pitch_var=2.0),
    'fearful':   dict(freq=240, mod=0.6, noise=0.07, pitch_var=1.8),
    'disgust':   dict(freq=160, mod=0.4, noise=0.05, pitch_var=0.8),
    'surprised': dict(freq=350, mod=0.8, noise=0.06, pitch_var=2.2),
}

def synthesize_emotion(emotion: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    p = EMOTION_PARAMS[emotion]
    t = np.linspace(0, duration, int(sr * duration))
    # Base tone + harmonic overtones
    sig  = np.sin(2 * np.pi * p['freq'] * t)
    sig += 0.5 * np.sin(2 * np.pi * p['freq'] * 2 * t)
    sig += 0.25 * np.sin(2 * np.pi * p['freq'] * 3 * t)
    # Amplitude modulation
    mod  = 1 + p['mod'] * np.sin(2 * np.pi * 4 * t)
    # Pitch variation (vibrato)
    pitch_shift = p['pitch_var'] * np.sin(2 * np.pi * 5 * t)
    sig_var = np.sin(2 * np.pi * (p['freq'] + pitch_shift) * t)
    sig = 0.6 * sig * mod + 0.4 * sig_var
    # Gaussian noise
    sig += p['noise'] * np.random.randn(len(t))
    # Normalize
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig.astype(np.float32)

print("🎙️  Generating synthetic audio clips …")
audio_data, labels = [], []
for emotion in EMOTIONS:
    for _ in range(SAMPLES_PER_EMOTION):
        sig = synthesize_emotion(emotion)
        # Add small random tempo variation
        stretch = np.random.uniform(0.9, 1.1)
        sig = librosa.effects.time_stretch(sig, rate=stretch)
        sig = librosa.util.fix_length(sig, size=int(SR * DURATION))
        audio_data.append(sig)
        labels.append(emotion)

print(f"   ✓ {len(audio_data)} clips | {len(EMOTIONS)} emotions")

# ═══════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════
N_MFCC   = 40
N_MELS   = 64
HOP      = 512
N_FFT    = 2048

def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return flat feature vector: MFCCs + delta + chroma + ZCR + RMS."""
    mfcc        = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma      = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP)
    zcr         = librosa.feature.zero_crossing_rate(signal, hop_length=HOP)
    rms         = librosa.feature.rms(y=signal, hop_length=HOP)

    feats = np.concatenate([
        np.mean(mfcc, axis=1),        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),  np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
        np.mean(chroma, axis=1),      np.std(chroma, axis=1),
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)],
    ])
    return feats

def extract_spectrogram(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return log-mel spectrogram shaped (n_mels, time_frames, 1)."""
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Fixed width
    target_w = 108
    if log_mel.shape[1] < target_w:
        log_mel = np.pad(log_mel, ((0,0),(0, target_w - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :target_w]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return log_mel[..., np.newaxis]          # (64, 108, 1)

print("🔬  Extracting features …")
X_flat, X_spec = [], []
for sig in audio_data:
    X_flat.append(extract_features(sig))
    X_spec.append(extract_spectrogram(sig))

X_flat = np.array(X_flat, dtype=np.float32)
X_spec = np.array(X_spec, dtype=np.float32)

le = LabelEncoder()
y_enc = le.fit_transform(labels)
y_cat = to_categorical(y_enc)
n_classes = len(EMOTIONS)
print(f"   ✓ Flat features: {X_flat.shape}  |  Spectrograms: {X_spec.shape}")

# ═══════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════
(Xf_tr, Xf_te, Xs_tr, Xs_te,
 y_tr,  y_te,  ye_tr, ye_te) = train_test_split(
    X_flat, X_spec, y_cat, y_enc,
    test_size=0.2, random_state=42, stratify=y_enc
)

# Normalise flat features
feat_mean = Xf_tr.mean(axis=0); feat_std = Xf_tr.std(axis=0) + 1e-9
Xf_tr = (Xf_tr - feat_mean) / feat_std
Xf_te = (Xf_te - feat_mean) / feat_std

# LSTM input: (samples, time_steps, features)
n_steps, n_feat_per_step = 20, X_flat.shape[1] // 20
if X_flat.shape[1] % 20 != 0:
    pad = 20 - (X_flat.shape[1] % 20)
    Xf_tr_rnn = np.pad(Xf_tr, ((0,0),(0,pad))).reshape(-1, 20, (X_flat.shape[1]+pad)//20)
    Xf_te_rnn = np.pad(Xf_te, ((0,0),(0,pad))).reshape(-1, 20, (X_flat.shape[1]+pad)//20)
else:
    Xf_tr_rnn = Xf_tr.reshape(-1, 20, X_flat.shape[1]//20)
    Xf_te_rnn = Xf_te.reshape(-1, 20, X_flat.shape[1]//20)

# ═══════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════
CB = [callbacks.EarlyStopping(patience=8, restore_best_weights=True),
      callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)]

# ── 4a. CNN on spectrograms ────────────────────────────
def build_cnn(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ── 4b. Bidirectional LSTM on flat features ───────────
def build_lstm(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ── 4c. Hybrid CNN-LSTM ───────────────────────────────
def build_hybrid(spec_shape, seq_shape, n_classes):
    # CNN branch
    si = layers.Input(shape=spec_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(si)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    cnn_out = layers.Dense(64, activation='relu')(x)
    # LSTM branch
    ri = layers.Input(shape=seq_shape)
    r = layers.LSTM(64, return_sequences=False)(ri)
    r = layers.Dropout(0.3)(r)
    lstm_out = layers.Dense(64, activation='relu')(r)
    # Merge
    merged = layers.Concatenate()([cnn_out, lstm_out])
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.4)(merged)
    out = layers.Dense(n_classes, activation='softmax')(merged)
    m = models.Model([si, ri], out)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ═══════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════
print("\n🧠  Training CNN …")
cnn = build_cnn(Xs_tr.shape[1:], n_classes)
hist_cnn = cnn.fit(Xs_tr, y_tr, validation_split=0.15,
                   epochs=60, batch_size=32, callbacks=CB, verbose=0)
print(f"   ✓ Best val-acc: {max(hist_cnn.history['val_accuracy']):.4f}")

print("🧠  Training Bidirectional LSTM …")
lstm = build_lstm(Xf_tr_rnn.shape[1:], n_classes)
hist_lstm = lstm.fit(Xf_tr_rnn, y_tr, validation_split=0.15,
                     epochs=60, batch_size=32, callbacks=CB, verbose=0)
print(f"   ✓ Best val-acc: {max(hist_lstm.history['val_accuracy']):.4f}")

print("🧠  Training Hybrid CNN-LSTM …")
hyb = build_hybrid(Xs_tr.shape[1:], Xf_tr_rnn.shape[1:], n_classes)
hist_hyb = hyb.fit([Xs_tr, Xf_tr_rnn], y_tr, validation_split=0.15,
                   epochs=60, batch_size=32, callbacks=CB, verbose=0)
print(f"   ✓ Best val-acc: {max(hist_hyb.history['val_accuracy']):.4f}")

# ═══════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════
def evaluate(name, model, X_in, y_true_cat, y_true_enc):
    prob = model.predict(X_in, verbose=0)
    pred = np.argmax(prob, axis=1)
    acc  = accuracy_score(y_true_enc, pred)
    f1   = f1_score(y_true_enc, pred, average='weighted')
    # ROC-AUC (one-vs-rest)
    y_bin = label_binarize(y_true_enc, classes=range(n_classes))
    auc_s = roc_auc_score(y_bin, prob, average='weighted', multi_class='ovr')
    return dict(name=name, acc=acc, f1=f1, auc=auc_s, prob=prob, pred=pred)

r_cnn  = evaluate('CNN',         cnn,  Xs_te,            y_te, ye_te)
r_lstm = evaluate('Bi-LSTM',     lstm, Xf_te_rnn,        y_te, ye_te)
r_hyb  = evaluate('CNN-LSTM',    hyb, [Xs_te, Xf_te_rnn],y_te, ye_te)
results = [r_cnn, r_lstm, r_hyb]

for r in results:
    print(f"  {r['name']:10s}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")

# ═══════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════
EMO_COLORS = ['#6366F1','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899','#14B8A6']
BG   = '#0F172A';  CARD = '#1E293B';  TEXT = '#F1F5F9'
GRID = '#334155';  ACC1 = '#6366F1';  ACC2 = '#10B981';  ACC3 = '#F59E0B'
MODEL_COLORS = [ACC1, ACC2, ACC3]

fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle('Emotion Recognition from Speech  ·  Deep Learning Pipeline',
             fontsize=24, fontweight='bold', color=TEXT, y=0.98, family='monospace')

gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97, top=0.95, bottom=0.03)

def card_ax(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return ax

# ── ROW 0: Sample waveform + MFCCs for 2 emotions ────
for col, emo in enumerate(['happy', 'angry', 'sad']):
    ax = card_ax(fig.add_subplot(gs[0, col]))
    sig = synthesize_emotion(emo)
    mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=13, hop_length=HOP)
    librosa.display.specshow(mfcc, sr=SR, hop_length=HOP, x_axis='time',
                             ax=ax, cmap='magma')
    ax.set_title(f'MFCC — {emo.capitalize()}', fontsize=11)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('MFCC Coefficient')

# ── ROW 1: Log-Mel Spectrograms ──────────────────────
for col, emo in enumerate(['fearful', 'disgust', 'surprised']):
    ax = card_ax(fig.add_subplot(gs[1, col]))
    sig = synthesize_emotion(emo)
    mel = librosa.feature.melspectrogram(y=sig, sr=SR, n_mels=N_MELS, hop_length=HOP)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_mel, sr=SR, hop_length=HOP,
                             x_axis='time', y_axis='mel',
                             ax=ax, cmap='viridis')
    ax.set_title(f'Mel-Spectrogram — {emo.capitalize()}', fontsize=11)

# ── ROW 2a: Training curves (all 3 models) ───────────
ax_tr = card_ax(fig.add_subplot(gs[2, :2]))
for (h, r, c) in [(hist_cnn, r_cnn, ACC1),(hist_lstm, r_lstm, ACC2),(hist_hyb, r_hyb, ACC3)]:
    ep = range(1, len(h.history['accuracy'])+1)
    ax_tr.plot(ep, h.history['accuracy'],     color=c, lw=2,   label=f"{r['name']} train")
    ax_tr.plot(ep, h.history['val_accuracy'], color=c, lw=1.5, linestyle='--', alpha=0.7)
ax_tr.set_title('Training & Validation Accuracy', fontsize=12)
ax_tr.set_xlabel('Epoch'); ax_tr.set_ylabel('Accuracy')
ax_tr.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT)
ax_tr.grid(color=GRID, alpha=0.4)
ax_tr.spines[['top','right']].set_visible(False)

# ── ROW 2b: Metric bar chart ──────────────────────────
ax_bar = card_ax(fig.add_subplot(gs[2, 2]))
metrics = ['acc', 'f1', 'auc']
xlabs   = ['Accuracy', 'F1-Score', 'ROC-AUC']
x = np.arange(3); w = 0.22
for i, (r, c) in enumerate(zip(results, MODEL_COLORS)):
    vals = [r[m] for m in metrics]
    bars = ax_bar.bar(x + i*w, vals, w, color=c, alpha=0.85, zorder=3)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, color=TEXT)
ax_bar.set_xticks(x + w); ax_bar.set_xticklabels(xlabs, fontsize=9)
ax_bar.set_ylim(0, 1.15); ax_bar.set_title('Model Metrics', fontsize=11)
ax_bar.yaxis.grid(True, color=GRID, alpha=0.4, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.spines[['top','right']].set_visible(False)
handles = [plt.Rectangle((0,0),1,1,color=c) for c in MODEL_COLORS]
ax_bar.legend(handles, [r['name'] for r in results], fontsize=8,
              facecolor=CARD, labelcolor=TEXT)

# ── ROW 3: Confusion matrices ─────────────────────────
for col, r in enumerate(results):
    ax_cm = card_ax(fig.add_subplot(gs[3, col]))
    cm = confusion_matrix(ye_te, r['pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_labels = [e[:4].capitalize() for e in EMOTIONS]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax_cm, cbar=False, annot_kws={'size': 8})
    ax_cm.set_title(f'{r["name"]}  Confusion Matrix\n(row-normalised)', fontsize=10)
    ax_cm.set_xlabel('Predicted', color=TEXT, fontsize=9)
    ax_cm.set_ylabel('Actual',    color=TEXT, fontsize=9)
    ax_cm.tick_params(axis='x', rotation=45)

# ── ROW 4a: Per-class F1 (best model) ────────────────
best_r = max(results, key=lambda r: r['f1'])
ax_f1  = card_ax(fig.add_subplot(gs[4, :2]))
report = classification_report(ye_te, best_r['pred'],
                                target_names=EMOTIONS, output_dict=True)
f1_per = [report[e]['f1-score'] for e in EMOTIONS]
bars   = ax_f1.bar(EMOTIONS, f1_per, color=EMO_COLORS, alpha=0.88, zorder=3)
for b, v in zip(bars, f1_per):
    ax_f1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
               f'{v:.2f}', ha='center', va='bottom', fontsize=9, color=TEXT)
ax_f1.set_ylim(0, 1.15)
ax_f1.set_title(f'Per-Emotion F1-Score  ({best_r["name"]} — best model)', fontsize=11)
ax_f1.set_ylabel('F1-Score'); ax_f1.set_xlabel('Emotion')
ax_f1.yaxis.grid(True, color=GRID, alpha=0.4, zorder=0)
ax_f1.set_axisbelow(True); ax_f1.spines[['top','right']].set_visible(False)
ax_f1.tick_params(axis='x', rotation=15)

# ── ROW 4b: ROC curves (best model, OvR) ─────────────
ax_roc = card_ax(fig.add_subplot(gs[4, 2]))
y_bin = label_binarize(ye_te, classes=range(n_classes))
for i, (emo, c) in enumerate(zip(EMOTIONS, EMO_COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], best_r['prob'][:, i])
    ax_roc.plot(fpr, tpr, color=c, lw=1.5,
                label=f"{emo[:5]} ({auc(fpr,tpr):.2f})")
ax_roc.plot([0,1],[0,1],'--', color=GRID, lw=1)
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.set_title(f'ROC Curves ({best_r["name"]})', fontsize=11)
ax_roc.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT)
ax_roc.grid(color=GRID, alpha=0.4)
ax_roc.spines[['top','right']].set_visible(False)

out_path = '/mnt/user-data/outputs/emotion_recognition_report.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n✅  Report saved → {out_path}")

# ── Final summary ─────────────────────────────────────
print("\n" + "═"*55)
print(f"  BEST MODEL : {best_r['name']}")
print(f"  Accuracy   : {best_r['acc']:.4f}")
print(f"  F1-Score   : {best_r['f1']:.4f}")
print(f"  ROC-AUC    : {best_r['auc']:.4f}")
print("═"*55)
print(classification_report(ye_te, best_r['pred'], target_names=EMOTIONS))