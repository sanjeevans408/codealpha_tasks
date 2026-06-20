"""
Credit Scoring Model
Predicts creditworthiness using Logistic Regression, Decision Tree, and Random Forest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Synthetic Dataset Generation
# ─────────────────────────────────────────────
np.random.seed(42)
N = 2000

age             = np.random.randint(21, 70, N)
income          = np.random.normal(55000, 20000, N).clip(15000, 200000)
debt_amount     = np.random.exponential(8000, N).clip(0, 80000)
credit_limit    = np.random.normal(15000, 5000, N).clip(500, 50000)
credit_util     = np.clip(debt_amount / credit_limit, 0, 1)
num_accounts    = np.random.randint(1, 15, N)
missed_payments = np.random.poisson(1, N).clip(0, 10)
months_employed = np.random.randint(0, 300, N)
num_inquiries   = np.random.poisson(1.5, N).clip(0, 10)
loan_amount     = np.random.normal(12000, 8000, N).clip(0, 80000)
education       = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], N,
                                    p=[0.35, 0.40, 0.18, 0.07])

# Feature engineering
debt_to_income      = debt_amount / (income + 1)
payment_history_score = np.clip(100 - missed_payments * 10, 0, 100)
employment_years    = months_employed / 12

# Credit score proxy (label generation with realistic logic)
credit_score = (
    0.30 * payment_history_score +
    0.25 * np.clip((1 - debt_to_income * 5) * 100, 0, 100) +
    0.15 * np.clip((1 - credit_util) * 100, 0, 100) +
    0.15 * np.clip(employment_years / 25 * 100, 0, 100) +
    0.10 * np.clip(income / 2000, 0, 100) +
    0.05 * np.clip(num_accounts / 15 * 100, 0, 100) +
    np.random.normal(0, 5, N)
)

# Binary label: 1 = creditworthy, 0 = not
target = (credit_score >= 50).astype(int)

edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
edu_encoded = np.array([edu_map[e] for e in education])

df = pd.DataFrame({
    'Age': age,
    'Income': income.round(2),
    'Debt_Amount': debt_amount.round(2),
    'Credit_Limit': credit_limit.round(2),
    'Credit_Utilization': credit_util.round(4),
    'Num_Accounts': num_accounts,
    'Missed_Payments': missed_payments,
    'Months_Employed': months_employed,
    'Num_Inquiries': num_inquiries,
    'Loan_Amount': loan_amount.round(2),
    'Education_Level': edu_encoded,
    'Debt_to_Income': debt_to_income.round(4),
    'Payment_History_Score': payment_history_score.round(2),
    'Employment_Years': employment_years.round(2),
    'Creditworthy': target
})

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Creditworthy'].value_counts()}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. Models
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
}

results = {}
for name, model in models.items():
    use_scaled = isinstance(model, LogisticRegression)
    Xtr = X_train_sc if use_scaled else X_train.values
    Xte = X_test_sc  if use_scaled else X_test.values

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    print(f"\n{name} — AUC: {results[name]['roc_auc']:.4f}  F1: {results[name]['f1']:.4f}")

# ─────────────────────────────────────────────
# 4. Visualization (single PDF-ready figure)
# ─────────────────────────────────────────────
PALETTE   = ['#2563EB', '#10B981', '#F59E0B']   # blue, green, amber
BG        = '#F8FAFC'
CARD      = '#FFFFFF'
TEXT      = '#1E293B'
GRIDCOLOR = '#E2E8F0'

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle('Credit Scoring Model — Performance Report',
             fontsize=22, fontweight='bold', color=TEXT, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38,
                       left=0.06, right=0.97, top=0.94, bottom=0.04)

model_names = list(results.keys())

# ── Row 0: Metric bar charts ──────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

ax_bar = fig.add_subplot(gs[0, :])
ax_bar.set_facecolor(CARD)
x       = np.arange(len(metrics))
width   = 0.22
for i, (name, color) in enumerate(zip(model_names, PALETTE)):
    vals = [results[name][m] for m in metrics]
    bars = ax_bar.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.88, zorder=3)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, color=TEXT)

ax_bar.set_xticks(x + width)
ax_bar.set_xticklabels(metric_labels, fontsize=11)
ax_bar.set_ylim(0, 1.12)
ax_bar.set_ylabel('Score', color=TEXT)
ax_bar.set_title('Model Performance Comparison', fontsize=13, color=TEXT, pad=10)
ax_bar.legend(loc='upper right', fontsize=9)
ax_bar.yaxis.grid(True, color=GRIDCOLOR, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.spines[['top', 'right']].set_visible(False)

# ── Row 1: ROC curves ────────────────────────
ax_roc = fig.add_subplot(gs[1, :2])
ax_roc.set_facecolor(CARD)
for (name, res), color in zip(results.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
ax_roc.set_xlabel('False Positive Rate', color=TEXT)
ax_roc.set_ylabel('True Positive Rate', color=TEXT)
ax_roc.set_title('ROC Curves', fontsize=13, color=TEXT)
ax_roc.legend(fontsize=9)
ax_roc.grid(color=GRIDCOLOR)
ax_roc.spines[['top', 'right']].set_visible(False)

# ── Row 1, col 2: Class distribution ─────────
ax_dist = fig.add_subplot(gs[1, 2])
ax_dist.set_facecolor(CARD)
counts = y.value_counts()
ax_dist.bar(['Not Creditworthy', 'Creditworthy'], counts.values,
            color=[PALETTE[2], PALETTE[0]], alpha=0.85, zorder=3)
for i, v in enumerate(counts.values):
    ax_dist.text(i, v + 15, str(v), ha='center', fontsize=10, color=TEXT)
ax_dist.set_title('Class Distribution', fontsize=13, color=TEXT)
ax_dist.yaxis.grid(True, color=GRIDCOLOR, zorder=0)
ax_dist.set_axisbelow(True)
ax_dist.spines[['top', 'right']].set_visible(False)

# ── Row 2: Confusion matrices ─────────────────
for col, (name, res) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs[2, col])
    ax_cm.set_facecolor(CARD)
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not CW', 'CW'],
                yticklabels=['Not CW', 'CW'],
                ax=ax_cm, cbar=False,
                annot_kws={'size': 13, 'weight': 'bold'})
    ax_cm.set_title(f'{name}\nConfusion Matrix', fontsize=10, color=TEXT)
    ax_cm.set_xlabel('Predicted', color=TEXT)
    ax_cm.set_ylabel('Actual', color=TEXT)

# ── Row 3, col 0–1: Feature importance (RF) ──
ax_fi = fig.add_subplot(gs[3, :2])
ax_fi.set_facecolor(CARD)
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True).tail(12)
colors_fi = [PALETTE[0] if v > feat_df['Importance'].median() else PALETTE[2]
             for v in feat_df['Importance']]
ax_fi.barh(feat_df['Feature'], feat_df['Importance'],
           color=colors_fi, alpha=0.85, zorder=3)
ax_fi.set_xlabel('Importance', color=TEXT)
ax_fi.set_title('Top Feature Importances (Random Forest)', fontsize=13, color=TEXT)
ax_fi.xaxis.grid(True, color=GRIDCOLOR, zorder=0)
ax_fi.set_axisbelow(True)
ax_fi.spines[['top', 'right']].set_visible(False)

# ── Row 3, col 2: Summary table ──────────────
ax_tbl = fig.add_subplot(gs[3, 2])
ax_tbl.axis('off')
tbl_data = []
for name, res in results.items():
    short = name.split()[0] if name != 'Logistic Regression' else 'LogReg'
    tbl_data.append([
        short,
        f"{res['accuracy']:.3f}",
        f"{res['precision']:.3f}",
        f"{res['recall']:.3f}",
        f"{res['f1']:.3f}",
        f"{res['roc_auc']:.3f}",
    ])
tbl = ax_tbl.table(
    cellText=tbl_data,
    colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC'],
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 2.0)
# Style header
for j in range(6):
    tbl[(0, j)].set_facecolor(PALETTE[0])
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
# Highlight best AUC row
best_idx = max(range(len(model_names)),
               key=lambda i: results[model_names[i]]['roc_auc'])
for j in range(6):
    tbl[(best_idx + 1, j)].set_facecolor('#DBEAFE')
ax_tbl.set_title('Summary (best AUC highlighted)', fontsize=10, color=TEXT, pad=12)

# Ensure a local outputs directory exists and save the report there (cross-platform)
output_dir = Path('outputs')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'credit_scoring_report.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n✅ Report saved to {output_path.resolve()}")

# ─────────────────────────────────────────────
# 5. Print classification report for best model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['roc_auc'])
print(f"\n{'='*55}")
print(f"Best Model: {best_name}")
print('='*55)
print(classification_report(y_test, results[best_name]['y_pred'],
                             target_names=['Not Creditworthy', 'Creditworthy']))