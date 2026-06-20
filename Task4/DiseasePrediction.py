# -*- coding: utf-8 -*-
"""
================================================================
  TASK 4 -- Disease Prediction from Medical Data
  CodeAlpha Machine Learning Internship
================================================================
  Datasets  : Heart Disease | Diabetes | Breast Cancer
  Algorithms: Logistic Regression | SVM | Random Forest | XGBoost
  Metrics   : Accuracy | Precision | Recall | F1 | AUC-ROC
================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import io
# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches
import seaborn as sns

from pathlib import Path
from itertools import cycle

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not found — skipping XGBoost model. Install with: pip install xgboost")

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────
PALETTE      = ["#4CC9F0", "#7B2FBE", "#F72585", "#4ADE80"]   # LR · SVM · RF · XGB
BG_COLOR     = "#0D1117"
PANEL_COLOR  = "#161B22"
TEXT_COLOR   = "#E6EDF3"
GRID_COLOR   = "#30363D"
ACCENT_COLOR = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    PANEL_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.5,
    "legend.facecolor":  PANEL_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
if not XGBOOST_AVAILABLE:
    MODEL_NAMES = MODEL_NAMES[:-1]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATASET GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_heart_disease(n: int = 1500) -> pd.DataFrame:
    """
    Synthetic Heart Disease dataset inspired by the UCI Cleveland dataset.
    Features reflect medically realistic distributions.
    """
    np.random.seed(42)
    age              = np.random.randint(29, 77,  n)
    sex              = np.random.choice([0, 1],    n, p=[0.32, 0.68])        # 0=F, 1=M
    chest_pain       = np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08])
    resting_bp       = np.random.normal(131, 17,   n).clip(94, 200).astype(int)
    cholesterol      = np.random.normal(246, 52,   n).clip(126, 564).astype(int)
    fasting_bs       = np.random.choice([0, 1],    n, p=[0.85, 0.15])
    resting_ecg      = np.random.choice([0, 1, 2], n, p=[0.52, 0.01, 0.47])
    max_hr           = np.random.normal(150, 23,   n).clip(71, 202).astype(int)
    exercise_angina  = np.random.choice([0, 1],    n, p=[0.67, 0.33])
    oldpeak          = np.random.exponential(1.0,  n).clip(0, 6.2).round(1)
    st_slope         = np.random.choice([0, 1, 2], n, p=[0.07, 0.60, 0.33])

    # Medically motivated label generation
    risk = (
        0.02 * (age - 29)                          # older → higher risk
        + 0.15 * sex                               # male → higher risk
        + 0.12 * (chest_pain == 0).astype(float)   # typical angina
        + 0.008 * (resting_bp - 94)
        + 0.003 * (cholesterol - 126)
        + 0.10 * fasting_bs
        + 0.12 * exercise_angina
        + 0.15 * oldpeak
        - 0.008 * (max_hr - 71)                    # higher max_hr → lower risk
        + 0.18 * (st_slope == 0).astype(float)     # flat/down slope
        + np.random.normal(0, 0.3, n)
    )
    target = (risk > np.percentile(risk, 46)).astype(int)   # ~54 % positive

    return pd.DataFrame({
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope,
        "HeartDisease": target,
    })


def generate_diabetes(n: int = 1500) -> pd.DataFrame:
    """
    Synthetic Diabetes dataset inspired by the Pima Indians Diabetes dataset.
    """
    np.random.seed(43)
    pregnancies       = np.random.poisson(3.8, n).clip(0, 17).astype(int)
    glucose           = np.random.normal(121, 32, n).clip(44, 199).astype(int)
    blood_pressure    = np.random.normal(69,  19, n).clip(24, 122).astype(int)
    skin_thickness    = np.random.normal(21,  16, n).clip(0,  99).astype(int)
    insulin           = np.random.exponential(80, n).clip(0, 846).astype(int)
    bmi               = np.random.normal(32,   7, n).clip(18,  67).round(1)
    dpf               = np.random.exponential(0.47, n).clip(0.08, 2.42).round(3)
    age               = np.random.randint(21, 81, n)

    risk = (
        0.025 * glucose
        + 0.04  * bmi
        + 0.02  * age
        + 0.05  * pregnancies
        + 0.005 * insulin
        + 0.30  * dpf
        - 0.01  * blood_pressure
        + np.random.normal(0, 0.8, n)
    )
    target = (risk > np.percentile(risk, 65)).astype(int)   # ~35 % positive

    return pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
        "Outcome": target,
    })


def generate_breast_cancer(n: int = 1500) -> pd.DataFrame:
    """
    Synthetic Breast Cancer dataset inspired by UCI Breast Cancer Wisconsin.
    Features are cell nucleus measurements.
    """
    np.random.seed(44)
    features = {}

    # Benign base stats (mean ± std) for each measurement
    benign_params = {
        "radius":           (12.1, 1.8), "texture":          (17.9, 4.0),
        "perimeter":        (78.1, 11.8), "area":             (462,  135),
        "smoothness":       (0.092, 0.014), "compactness":    (0.080, 0.034),
        "concavity":        (0.046, 0.044), "concave_points": (0.026, 0.016),
        "symmetry":         (0.174, 0.028), "fractal_dim":    (0.062, 0.007),
    }
    # Malignant base stats
    malignant_params = {
        "radius":           (17.5, 3.4), "texture":          (21.6, 4.2),
        "perimeter":        (115,  22.0), "area":             (978,  367),
        "smoothness":       (0.103, 0.014), "compactness":   (0.145, 0.054),
        "concavity":        (0.161, 0.090), "concave_points": (0.088, 0.036),
        "symmetry":         (0.193, 0.030), "fractal_dim":   (0.063, 0.008),
    }

    # Assign labels first (37 % malignant)
    labels = np.random.choice([0, 1], n, p=[0.63, 0.37])

    for feat, (mu_b, sd_b) in benign_params.items():
        mu_m, sd_m = malignant_params[feat]
        vals = np.where(
            labels == 0,
            np.random.normal(mu_b, sd_b, n),
            np.random.normal(mu_m, sd_m, n),
        ).clip(0)
        features[f"{feat}_mean"]  = vals.round(4)
        features[f"{feat}_worst"] = (vals * np.random.uniform(1.0, 1.6, n)).round(4)

    features["Diagnosis"] = labels   # 1 = Malignant, 0 = Benign

    # Add small noise to make problem non-trivial
    df = pd.DataFrame(features)
    noise_cols = [c for c in df.columns if c != "Diagnosis"]
    df[noise_cols] += np.random.normal(0, 0.01, (n, len(noise_cols)))
    df[noise_cols] = df[noise_cols].clip(0)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODELLING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def build_models():
    """Return dict of unfitted classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        "SVM":                 SVC(kernel="rbf", C=1.0, probability=True),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42, verbosity=0,
        )
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names,
                       dataset_name: str):
    """
    Train all models, compute metrics, return results dict and fitted RF
    for feature-importance plotting.
    """
    # Scalers for distance-based models
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results   = {}
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted_rf = None

    print(f"\n{'='*64}")
    print(f"  {dataset_name}  ({len(X_train)+len(X_test)} samples, "
          f"{X_train.shape[1]} features)")
    print(f"{'='*64}")
    header = f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    print(header)
    print(f"  {'-'*62}")

    for name, clf in build_models().items():
        needs_scaling = name in ("Logistic Regression", "SVM")
        Xtr = X_train_sc if needs_scaling else X_train
        Xte = X_test_sc  if needs_scaling else X_test

        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)
        auc  = roc_auc_score  (y_test, y_prob)

        # Cross-validated AUC
        cv_auc = cross_val_score(
            clf, Xtr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        ).mean()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm          = confusion_matrix(y_test, y_pred)

        results[name] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "auc": auc, "cv_auc": cv_auc,
            "fpr": fpr, "tpr": tpr, "cm": cm,
            "clf": clf,
        }

        print(f"  {name:<22} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} "
              f"{f1:>6.3f} {auc:>6.3f}  (CV AUC={cv_auc:.3f})")

        if name == "Random Forest":
            fitted_rf = clf

    # Determine best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  [BEST] {best_name} (AUC={results[best_name]['auc']:.3f})")

    return results, fitted_rf, best_name


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _title_ax(ax, title: str, fontsize: int = 12):
    ax.set_title(title, fontsize=fontsize, fontweight="bold",
                 color=TEXT_COLOR, pad=10)


def plot_class_distribution(ax, y, class_names, color_pair):
    """Donut chart showing class balance."""
    counts  = [np.sum(y == 0), np.sum(y == 1)]
    explode = (0.04, 0.04)
    wedges, texts, autotexts = ax.pie(
        counts, labels=class_names, autopct="%1.1f%%",
        colors=color_pair, startangle=90,
        explode=explode, wedgeprops=dict(width=0.55, edgecolor=BG_COLOR, linewidth=2),
        textprops=dict(color=TEXT_COLOR, fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(BG_COLOR)
        at.set_fontweight("bold")
    _title_ax(ax, "Class Distribution")


def plot_correlation_heatmap(ax, df, target_col):
    """Heatmap of top-10 correlated features with target."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr        = numeric_df.corr()
    target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_feats   = target_corr.head(10).index.tolist()
    sub_corr    = numeric_df[top_feats + [target_col]].corr()

    mask = np.zeros_like(sub_corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        sub_corr, ax=ax, mask=mask, cmap=cmap,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.5, linecolor=BG_COLOR,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
    _title_ax(ax, "Feature Correlation (Top-10)")


def plot_model_accuracy(ax, results):
    """Grouped bar chart — accuracy, F1, AUC per model."""
    names   = list(results.keys())
    x       = np.arange(len(names))
    metrics = {"Accuracy": [results[n]["acc"]  for n in names],
               "F1 Score": [results[n]["f1"]   for n in names],
               "AUC-ROC":  [results[n]["cv_auc"] for n in names]}
    width   = 0.25
    offsets = [-width, 0, width]
    bar_colors = ["#4CC9F0", "#F72585", "#4ADE80"]

    for (metric, vals), off, col in zip(metrics.items(), offsets, bar_colors):
        bars = ax.bar(x + off, vals, width, label=metric,
                      color=col, alpha=0.85, zorder=3,
                      edgecolor=BG_COLOR, linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=TEXT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    _title_ax(ax, "Model Performance Comparison")


def plot_roc_curves(ax, results):
    """Overlaid ROC curves for all models."""
    colors = dict(zip(results.keys(), PALETTE))
    for name, r in results.items():
        auc_val = r["cv_auc"]
        ax.plot(r["fpr"], r["tpr"], lw=2, color=colors[name],
                label=f"{name} (AUC={auc_val:.3f})", alpha=0.9)

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random (AUC=0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="white")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    _title_ax(ax, "ROC Curves — All Models")


def plot_confusion_matrix(ax, results, best_name, class_names):
    """Confusion matrix for the best model."""
    cm = results[best_name]["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", [PANEL_COLOR, ACCENT_COLOR], N=256
    )
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = BG_COLOR if cm_norm[i, j] > thresh else TEXT_COLOR
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    _title_ax(ax, f"Confusion Matrix — {best_name}")


def plot_feature_importance(ax, rf_model, feature_names, top_n=12):
    """Horizontal bar chart of RF feature importances."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[-top_n:]
    feats       = [feature_names[i] for i in indices]
    vals        = importances[indices]

    norm  = plt.Normalize(vals.min(), vals.max())
    cmap  = matplotlib.colormaps.get_cmap("cool")
    colors = [cmap(norm(v)) for v in vals]

    bars = ax.barh(range(len(feats)), vals, color=colors, edgecolor=BG_COLOR,
                   linewidth=0.5, zorder=3)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Importance", fontsize=9)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, color=TEXT_COLOR)
    _title_ax(ax, "Feature Importance — Random Forest")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PER-DATASET FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def create_disease_figure(df, target_col, class_names, dataset_name,
                          color_pair, output_path):
    """
    Generate and save a 2×3 panel figure for one disease dataset.
    """
    # ── Prepare data ──────────────────────────────────────────────────────────
    X       = df.drop(columns=[target_col])
    y       = df[target_col].values
    feat_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.20, random_state=42, stratify=y
    )

    results, fitted_rf, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_names, dataset_name
    )

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor=BG_COLOR)

    # Title banner
    fig.text(0.5, 0.97, dataset_name, ha="center", va="top",
             fontsize=22, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.94,
             "Classification Algorithms: Logistic Regression · SVM · Random Forest · XGBoost",
             ha="center", va="top", fontsize=11, color="#8B949E")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97, top=0.92, bottom=0.06,
                           hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_class_distribution(ax1, y, class_names, color_pair)
    plot_correlation_heatmap(ax2, df, target_col)
    plot_model_accuracy(ax3, results)
    plot_roc_curves(ax4, results)
    plot_confusion_matrix(ax5, results, best_name, class_names)
    plot_feature_importance(ax6, fitted_rf, feat_names)

    fig.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"\n  [SAVED] {output_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-DATASET SUMMARY FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def create_summary_figure(all_results: dict):
    """
    Comparative summary: grouped bars + heatmap across all 3 diseases.
    """
    dataset_names = list(all_results.keys())
    metric_keys   = ["acc", "prec", "rec", "f1", "cv_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC\n(CV)"]

    fig = plt.figure(figsize=(20, 13), facecolor=BG_COLOR)
    fig.text(0.5, 0.97, "Disease Prediction — Cross-Dataset Model Summary",
             ha="center", va="top", fontsize=21, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.94,
             "Comparative performance across Heart Disease · Diabetes · Breast Cancer",
             ha="center", va="top", fontsize=11, color="#8B949E")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97, top=0.91, bottom=0.06,
                           hspace=0.45, wspace=0.32)

    # ── Panel A: AUC per model across datasets ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    model_keys = list(next(iter(all_results.values())).keys())
    x      = np.arange(len(dataset_names))
    width  = 0.20
    n_m    = len(model_keys)
    offsets = np.linspace(-width * (n_m-1)/2, width * (n_m-1)/2, n_m)

    for (model, off, col) in zip(model_keys, offsets, PALETTE):
        auc_vals = [all_results[ds][model]["cv_auc"] for ds in dataset_names]
        bars = ax_a.bar(x + off, auc_vals, width * 0.9, label=model,
                        color=col, alpha=0.88, zorder=3,
                        edgecolor=BG_COLOR, linewidth=0.5)
        for bar, val in zip(bars, auc_vals):
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=7.5, color=TEXT_COLOR, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(dataset_names, fontsize=11)
    ax_a.set_ylabel("5-Fold CV AUC-ROC", fontsize=10)
    ax_a.set_ylim(0, 1.12)
    ax_a.legend(fontsize=9, loc="lower right", ncol=len(model_keys))
    ax_a.yaxis.grid(True, alpha=0.4)
    ax_a.set_axisbelow(True)
    _title_ax(ax_a, "AUC-ROC Comparison — All Models × All Datasets", fontsize=13)

    # ── Panel B: Performance heatmap (AUC) ────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    heat_data = pd.DataFrame(
        {ds: {m: all_results[ds][m]["cv_auc"] for m in model_keys}
         for ds in dataset_names}
    )
    cmap_heat = sns.light_palette(ACCENT_COLOR, as_cmap=True)
    sns.heatmap(heat_data, ax=ax_b, cmap=cmap_heat,
                annot=True, fmt=".3f", annot_kws={"size": 11, "fontweight": "bold"},
                vmin=0.5, vmax=1.0,
                linewidths=1, linecolor=BG_COLOR,
                cbar_kws={"label": "AUC-ROC"})
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0,  fontsize=9)
    _title_ax(ax_b, "AUC Heatmap (Models × Datasets)")

    # ── Panel C: Radar / spider chart for best-model metrics ─────────────
    ax_c = fig.add_subplot(gs[1, 1], polar=True)
    angles  = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    ds_colors = ["#4CC9F0", "#F72585", "#4ADE80"]
    for ds, col in zip(dataset_names, ds_colors):
        best_m = max(all_results[ds], key=lambda k: all_results[ds][k]["cv_auc"])
        vals   = [all_results[ds][best_m][mk] for mk in metric_keys]
        vals  += vals[:1]
        ax_c.plot(angles, vals, lw=2, color=col, label=f"{ds}\n({best_m})")
        ax_c.fill(angles, vals, alpha=0.12, color=col)

    ax_c.set_xticks(angles[:-1])
    ax_c.set_xticklabels(metric_labels, size=9, color=TEXT_COLOR)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_c.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="#8B949E")
    ax_c.grid(color=GRID_COLOR, linewidth=0.6)
    ax_c.spines["polar"].set_color(GRID_COLOR)
    ax_c.set_facecolor(PANEL_COLOR)
    ax_c.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                fontsize=8, framealpha=0.3)
    _title_ax(ax_c, "Best-Model Metrics Radar\n(per Disease)", fontsize=11)

    out = OUTPUT_DIR / "model_comparison_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"\n  [SAVED] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  DISEASE PREDICTION FROM MEDICAL DATA")
    print("  CodeAlpha ML Internship -- Task 4")
    print("=" * 64)

    all_results = {}

    # ── 7a. Heart Disease ─────────────────────────────────────────────────
    print("\n[1/3] Generating Heart Disease dataset …")
    df_heart = generate_heart_disease(n=1500)
    results_heart = create_disease_figure(
        df            = df_heart,
        target_col    = "HeartDisease",
        class_names   = ["No Disease", "Heart Disease"],
        dataset_name  = "Heart Disease Prediction",
        color_pair    = ("#F72585", "#4CC9F0"),
        output_path   = OUTPUT_DIR / "heart_disease_results.png",
    )
    all_results["Heart Disease"] = results_heart

    # ── 7b. Diabetes ──────────────────────────────────────────────────────
    print("\n[2/3] Generating Diabetes dataset …")
    df_diabetes = generate_diabetes(n=1500)
    results_diabetes = create_disease_figure(
        df            = df_diabetes,
        target_col    = "Outcome",
        class_names   = ["Non-Diabetic", "Diabetic"],
        dataset_name  = "Diabetes Prediction",
        color_pair    = ("#4ADE80", "#7B2FBE"),
        output_path   = OUTPUT_DIR / "diabetes_results.png",
    )
    all_results["Diabetes"] = results_diabetes

    # ── 7c. Breast Cancer ─────────────────────────────────────────────────
    print("\n[3/3] Generating Breast Cancer dataset …")
    df_cancer = generate_breast_cancer(n=1500)
    results_cancer = create_disease_figure(
        df            = df_cancer,
        target_col    = "Diagnosis",
        class_names   = ["Benign", "Malignant"],
        dataset_name  = "Breast Cancer Prediction",
        color_pair    = ("#F72585", "#4ADE80"),
        output_path   = OUTPUT_DIR / "breast_cancer_results.png",
    )
    all_results["Breast Cancer"] = results_cancer

    # ── 7d. Cross-dataset summary ─────────────────────────────────────────
    print("\n[Summary] Generating cross-dataset comparison figure...")
    create_summary_figure(all_results)

    # ── 7e. Final console report ──────────────────────────────────────────
    # -- 7e. Final console report ------------------------------------------
    print("\n" + "=" * 64)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 64)
    for ds_name, res in all_results.items():
        print(f"\n  {ds_name}")
        print(f"  {'-'*42}")
        print(f"  {'Model':<22} {'AUC(CV)':>8} {'F1':>8} {'Acc':>8}")
        print(f"  {'-'*42}")
        for m, r in res.items():
            print(f"  {m:<22} {r['cv_auc']:>8.4f} {r['f1']:>8.4f} {r['acc']:>8.4f}")

    print("\n" + "=" * 64)
    print(f"  [OK] All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
