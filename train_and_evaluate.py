#!/usr/bin/env python3
# train_and_evaluate.py
"""
Train and evaluate ML models for network traffic classification.
- Input: ~/tma_project/pcaps_csv/combined_features.csv
- Output: models, scaler, plots, metrics in ~/tma_project/results/

Improvements:
- Better handling of missing values
- Class imbalance handling
- More robust feature selection
- Better visualization
- Configurable split strategy (time-aware vs random)
- Save predictions for analysis
- Fixed plot_class_distribution function
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

warnings.filterwarnings('ignore')

# -----------------------
# ⚙️ CONFIGURATION - CHANGE THIS!
# -----------------------
TIME_AWARE_SPLIT = False  # ✅ Set to False for random split, True for time-based split
RANDOM_STATE = 42        # For reproducibility

# -----------------------
# Config / paths
# -----------------------
BASE_DIR = os.path.expanduser('~/tma_project')
CSV_PATH = os.path.join(BASE_DIR, 'pcaps_csv', 'combined_features.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_TXT = os.path.join(RESULTS_DIR, 'metrics.txt')

# Clear previous metrics file
if os.path.exists(METRICS_TXT):
    os.remove(METRICS_TXT)

# -----------------------
# Utility functions
# -----------------------
def save_metrics(text):
    """Append metrics to log file."""
    with open(METRICS_TXT, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")
    print(text)

def plot_and_save_cm(y_true, y_pred, title, fname):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    return cm

def plot_and_save_roc(y_true, y_scores, title, fname):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    return roc_auc

def plot_class_distribution(y, title, fname):
    """Plot class distribution - FIXED VERSION."""
    plt.figure(figsize=(6,4))
    counts = y.value_counts()
    
    # Extract values safely
    normal_count = counts.get(0, 0)
    attack_count = counts.get(1, 0)
    values = [normal_count, attack_count]
    max_value = max(values) if values else 1
    
    plt.bar(['Normal', 'Attack'], values, color=['#2ecc71', '#e74c3c'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    
    # Add text labels
    for i, v in enumerate(values):
        plt.text(i, v + max_value*0.02, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()

# -----------------------
# Load data
# -----------------------
print("="*60)
print("NETWORK TRAFFIC ML CLASSIFICATION")
print("="*60)

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Input CSV not found: {CSV_PATH}\n"
        f"Make sure you have created combined_features.csv in ~/tma_project/pcaps_csv/"
    )

print(f"\n[1/8] Loading data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")

# Find time column
time_col = None
for candidate in ['frame.time_epoch', 'time_epoch', 'timestamp', 'time']:
    if candidate in df.columns:
        time_col = candidate
        break

if TIME_AWARE_SPLIT and time_col:
    df = df.sort_values(time_col).reset_index(drop=True)
    print(f"   ✓ Sorted by time column: {time_col}")
elif TIME_AWARE_SPLIT and not time_col:
    print("   ⚠ Warning: TIME_AWARE_SPLIT=True but no time column found!")
    print("   → Falling back to random split")
    TIME_AWARE_SPLIT = False

# -----------------------
# Prepare features and label
# -----------------------
print("\n[2/8] Preparing features and labels...")

if 'label' not in df.columns:
    raise KeyError("Input CSV must contain a 'label' column with 0 (normal) / 1 (attack).")

# Identify columns to exclude
exclude_cols = ['label']
if time_col:
    exclude_cols.append(time_col)

# Get numeric columns only
numeric_df = df.select_dtypes(include=[np.number]).copy()
for ex in exclude_cols:
    if ex in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[ex])

print(f"   Found {len(numeric_df.columns)} numeric features")

# Handle missing values
missing_before = numeric_df.isnull().sum().sum()
if missing_before > 0:
    print(f"   ⚠ Found {missing_before} missing values, filling with median...")
    numeric_df = numeric_df.fillna(numeric_df.median())

# Handle infinite values
inf_mask = np.isinf(numeric_df).any(axis=1)
if inf_mask.sum() > 0:
    print(f"   ⚠ Found {inf_mask.sum()} rows with infinite values, replacing...")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median())

# Remove constant features (no variance)
constant_features = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
if constant_features:
    print(f"   Removing {len(constant_features)} constant features")
    numeric_df = numeric_df.drop(columns=constant_features)

# Final features and labels
X_all = numeric_df.copy()
y_all = df['label'].astype(int).reset_index(drop=True)

print(f"   Final feature shape: {X_all.shape}")
print(f"   Label distribution:")
label_counts = y_all.value_counts()
for label, count in label_counts.items():
    label_name = "Normal" if label == 0 else "Attack"
    percentage = (count / len(y_all)) * 100
    print(f"      {label_name} ({label}): {count} ({percentage:.1f}%)")

# Plot overall class distribution
plot_class_distribution(y_all, 'Overall Class Distribution', 'class_distribution.png')

# -----------------------
# Train/Val/Test split (60/20/20)
# -----------------------
print(f"\n[3/8] Splitting data ({'TIME-AWARE' if TIME_AWARE_SPLIT else 'RANDOM'}: 60% train, 20% val, 20% test)...")

if TIME_AWARE_SPLIT:
    # Time-based sequential split
    n = len(df)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train_idx = range(0, n_train)
    val_idx = range(n_train, n_train + n_val)
    test_idx = range(n_train + n_val, n)

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    X_val = X_all.iloc[val_idx].reset_index(drop=True)
    y_val = y_all.iloc[val_idx].reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True)
    
    print("   ⏰ Using time-aware sequential split")
else:
    # Random stratified split
    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, 
        test_size=0.4, 
        random_state=RANDOM_STATE,
        stratify=y_all
    )
    
    # Second split: split temp into 50-50 (which gives 20% val, 20% test of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    print("   🎲 Using random stratified split")

print(f"   Train: {len(X_train)} samples")
print(f"   Val:   {len(X_val)} samples")
print(f"   Test:  {len(X_test)} samples")

# Check class distribution in each split
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    counts = y_split.value_counts()
    normal = counts.get(0, 0)
    attack = counts.get(1, 0)
    total = len(y_split)
    normal_pct = (normal/total*100) if total > 0 else 0
    attack_pct = (attack/total*100) if total > 0 else 0
    print(f"   {split_name:5} distribution: Normal={normal:5} ({normal_pct:5.1f}%), Attack={attack:5} ({attack_pct:5.1f}%)")

# Warning if imbalanced splits
if TIME_AWARE_SPLIT:
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        counts = y_split.value_counts()
        if len(counts) < 2:
            print(f"   ⚠️  WARNING: {split_name} set contains only ONE class!")
            print(f"      → Consider using TIME_AWARE_SPLIT = False for better balance")

# -----------------------
# Scaling
# -----------------------
print("\n[4/8] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"   ✓ Scaler saved to: {scaler_path}")

# -----------------------
# Train models
# -----------------------
print("\n[5/8] Training models...")

# Random Forest
print("   Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)
rf.fit(X_train, y_train)
rf_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
joblib.dump(rf, rf_path)
print(f"   ✓ RandomForest saved to: {rf_path}")

# KNN
print("   Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
knn_path = os.path.join(MODELS_DIR, 'knn_model.joblib')
joblib.dump(knn, knn_path)
print(f"   ✓ KNN saved to: {knn_path}")

# -----------------------
# Evaluation function
# -----------------------
def evaluate_and_report(model, X, y, X_scaled, name="model", dataset="test"):
    """Evaluate model and generate comprehensive report."""
    print(f"\n   Evaluating {name} on {dataset} set...")
    
    # Use scaled or unscaled based on model
    if 'KNN' in name:
        X_infer = X_scaled
    else:
        X_infer = X
    
    # Predictions
    y_pred = model.predict(X_infer)
    
    # Get probability scores
    try:
        y_score = model.predict_proba(X_infer)[:,1]
    except Exception:
        y_score = y_pred
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y, y_score)
    except Exception:
        roc_auc = 0.0
    
    # Print metrics
    metrics_text = (
        f"{name} ({dataset}): "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, "
        f"Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
    )
    save_metrics(metrics_text)
    
    # Confusion matrix
    cm = plot_and_save_cm(y, y_pred, 
                          f'{name} - {dataset.capitalize()} Set',
                          f'{name}_{dataset}_cm.png')
    
    # ROC curve
    try:
        auc_val = plot_and_save_roc(y, y_score, 
                                     f'{name} ROC - {dataset.capitalize()} Set',
                                     f'{name}_{dataset}_roc.png')
    except Exception as e:
        print(f"      ⚠ ROC plotting failed: {e}")
        auc_val = roc_auc
    
    # Classification report
    cr = classification_report(y, y_pred, 
                               target_names=['Normal', 'Attack'],
                               zero_division=0)
    report_path = os.path.join(RESULTS_DIR, f"{name}_{dataset}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{name} - {dataset.capitalize()} Set\n")
        f.write("="*50 + "\n\n")
        f.write(cr)
        f.write(f"\n\nConfusion Matrix:\n{cm}\n")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': y.values,
        'predicted_label': y_pred,
        'prediction_score': y_score
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, f'{name}_{dataset}_predictions.csv'), 
                   index=False)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }

# -----------------------
# Run evaluation
# -----------------------
print("\n[6/8] Evaluating models...")

# Evaluate on validation set first
print("\n--- Validation Set Evaluation ---")
rf_val_metrics = evaluate_and_report(rf, X_val, y_val, X_val_scaled, 
                                     name='RandomForest', dataset='validation')
knn_val_metrics = evaluate_and_report(knn, X_val, y_val, X_val_scaled, 
                                      name='KNN', dataset='validation')

# Evaluate on test set
print("\n--- Test Set Evaluation ---")
rf_test_metrics = evaluate_and_report(rf, X_test, y_test, X_test_scaled, 
                                      name='RandomForest', dataset='test')
knn_test_metrics = evaluate_and_report(knn, X_test, y_test, X_test_scaled, 
                                       name='KNN', dataset='test')

# -----------------------
# Feature importance (RF)
# -----------------------
print("\n[7/8] Analyzing feature importance...")

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_feats = feat_importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10,8))
top_feats.plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances (RandomForest)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_top20.png'), dpi=150)
plt.close()

# Save all feature importances
all_feats = feat_importances.sort_values(ascending=False)
all_feats.to_csv(os.path.join(RESULTS_DIR, 'all_feature_importances.csv'), 
                 header=['importance'])
print(f"   ✓ Feature importance saved")

# -----------------------
# Model comparison plot
# -----------------------
print("\n[8/8] Creating comparison plots...")

models_comparison = pd.DataFrame({
    'Model': ['RandomForest', 'KNN'],
    'Accuracy': [rf_test_metrics['accuracy'], knn_test_metrics['accuracy']],
    'Precision': [rf_test_metrics['precision'], knn_test_metrics['precision']],
    'Recall': [rf_test_metrics['recall'], knn_test_metrics['recall']],
    'F1-Score': [rf_test_metrics['f1'], knn_test_metrics['f1']],
    'ROC-AUC': [rf_test_metrics['roc_auc'], knn_test_metrics['roc_auc']]
})

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models_comparison['Model']))
width = 0.15
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, models_comparison[metric], width, label=metric)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison (Test Set)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(models_comparison['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# Save comparison table
models_comparison.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), 
                        index=False)

# -----------------------
# Final summary
# -----------------------
split_method = "Time-aware Sequential" if TIME_AWARE_SPLIT else "Random Stratified"
summary_text = f"""
{'='*60}
FINAL RESULTS SUMMARY
{'='*60}

Configuration:
- Split Method: {split_method}
- Random State: {RANDOM_STATE}

Dataset Statistics:
- Total samples: {len(df)}
- Training samples: {len(X_train)}
- Validation samples: {len(X_val)}
- Test samples: {len(X_test)}
- Number of features: {X_train.shape[1]}

RandomForest (Test Set):
- Accuracy:  {rf_test_metrics['accuracy']:.4f}
- Precision: {rf_test_metrics['precision']:.4f}
- Recall:    {rf_test_metrics['recall']:.4f}
- F1-Score:  {rf_test_metrics['f1']:.4f}
- ROC-AUC:   {rf_test_metrics['roc_auc']:.4f}

KNN (Test Set):
- Accuracy:  {knn_test_metrics['accuracy']:.4f}
- Precision: {knn_test_metrics['precision']:.4f}
- Recall:    {knn_test_metrics['recall']:.4f}
- F1-Score:  {knn_test_metrics['f1']:.4f}
- ROC-AUC:   {knn_test_metrics['roc_auc']:.4f}

Files saved in: {RESULTS_DIR}
Models saved in: {MODELS_DIR}
Plots saved in: {PLOTS_DIR}

{'='*60}
"""

print(summary_text)

with open(os.path.join(RESULTS_DIR, 'summary_report.txt'), 'w') as f:
    f.write(summary_text)

save_metrics(f"Training completed using {split_method} split!")

print(f"\n✓ All done! Check results in: {RESULTS_DIR}")
