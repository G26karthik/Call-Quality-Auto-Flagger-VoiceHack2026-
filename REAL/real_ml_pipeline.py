"""
REAL CareCaller ML Pipeline
Extreme imbalance handling (1.46% positive rate)
Target: Maximize recall while keeping reasonable precision
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    precision_recall_curve, classification_report,
    roc_auc_score
)

# Try importing optional packages
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, will skip")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, will skip")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("SMOTE not available, will skip")

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

DATA_DIR = r"C:\Users\saita\OneDrive\Desktop\Projects\Caller.ai\REAL"

train_df = pd.read_csv(f"{DATA_DIR}/hackathon_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/hackathon_val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/hackathon_test.csv")

print(f"Train: {len(train_df)} rows, {train_df['has_ticket'].sum()} tickets ({100*train_df['has_ticket'].mean():.2f}%)")
print(f"Val:   {len(val_df)} rows, {val_df['has_ticket'].sum()} tickets ({100*val_df['has_ticket'].mean():.2f}%)")
print(f"Test:  {len(test_df)} rows")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

def engineer_features(df):
    """Create all features for the model"""
    features = pd.DataFrame()
    features['call_id'] = df['call_id']
    
    # --- Structural Features ---
    features['call_duration'] = df['call_duration'].fillna(0)
    features['answered_count'] = df['answered_count'].fillna(0)
    features['response_completeness'] = df['response_completeness'].fillna(0)
    features['pipeline_mismatch_count'] = df['pipeline_mismatch_count'].fillna(0)
    features['attempt_number'] = df['attempt_number'].fillna(1)
    features['form_submitted'] = df['form_submitted'].fillna(False).astype(int)
    features['interruption_count'] = df['interruption_count'].fillna(0)
    features['turn_count'] = df['turn_count'].fillna(0)
    features['user_word_count'] = df['user_word_count'].fillna(0)
    features['agent_word_count'] = df['agent_word_count'].fillna(0)
    features['avg_user_turn_words'] = df['avg_user_turn_words'].fillna(0)
    features['avg_agent_turn_words'] = df['avg_agent_turn_words'].fillna(0)
    features['question_count'] = df['question_count'].fillna(0)
    features['hour_of_day'] = df['hour_of_day'].fillna(12)
    features['day_of_week'] = df['day_of_week'].fillna(0)
    features['user_turn_count'] = df['user_turn_count'].fillna(0)
    features['agent_turn_count'] = df['agent_turn_count'].fillna(0)
    features['max_time_in_call'] = df['max_time_in_call'].fillna(0)
    features['billing_duration'] = df['billing_duration'].fillna(0)
    
    # --- Outcome Label Encoding ---
    outcome_map = {
        'completed': 0,
        'incomplete': 1,
        'wrong_number': 2,
        'voicemail': 3,
        'no_answer': 4,
        'busy': 5,
        'failed': 6
    }
    features['outcome_encoded'] = df['outcome'].map(outcome_map).fillna(7)
    
    # --- Engineered Features ---
    features['duration_per_answer'] = features['call_duration'] / (features['answered_count'] + 1)
    features['is_completed'] = (df['outcome'] == 'completed').astype(int)
    features['is_incomplete'] = (df['outcome'] == 'incomplete').astype(int)
    features['is_wrong_number'] = (df['outcome'] == 'wrong_number').astype(int)
    features['answer_gap'] = features['question_count'] - features['answered_count']
    features['high_duration'] = (features['call_duration'] > 150).astype(int)
    
    # Additional engineered features
    features['words_per_second'] = features['user_word_count'] / (features['call_duration'] + 1)
    features['agent_dominance'] = features['agent_word_count'] / (features['user_word_count'] + 1)
    features['completion_rate'] = features['answered_count'] / (features['question_count'] + 1)
    features['turn_density'] = features['turn_count'] / (features['call_duration'] + 1)
    features['long_call'] = (features['call_duration'] > 120).astype(int)
    features['very_long_call'] = (features['call_duration'] > 200).astype(int)
    features['high_answered'] = (features['answered_count'] > 5).astype(int)
    features['has_mismatch'] = (features['pipeline_mismatch_count'] > 0).astype(int)
    features['duration_answered_interaction'] = features['call_duration'] * features['answered_count']
    
    # --- Text Features from validation_notes ---
    notes = df['validation_notes'].fillna('').astype(str).str.lower()
    features['has_outcome_correction'] = (
        notes.str.contains('corrected call_outcome', regex=False) |
        notes.str.contains('outcome corrected', regex=False)
    ).astype(int)
    features['has_labeled_but'] = notes.str.contains(r'labeled outcome as.*but', regex=True).astype(int)
    features['notes_length'] = df['validation_notes'].fillna('').str.len()
    features['has_notes'] = (features['notes_length'] > 0).astype(int)
    
    # Pipeline status features
    features['review_flag'] = df['review_flag'].fillna(False).astype(int)
    
    return features

# Engineer features for all datasets
train_features = engineer_features(train_df)
val_features = engineer_features(val_df)
test_features = engineer_features(test_df)

# Define feature columns (exclude call_id)
feature_cols = [c for c in train_features.columns if c != 'call_id']
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols[:10]}... (showing first 10)")

# Prepare data
X_train = train_features[feature_cols].values
y_train = train_df['has_ticket'].astype(int).values

X_val = val_features[feature_cols].values
y_val = val_df['has_ticket'].astype(int).values

X_test = test_features[feature_cols].values

# Combine train + val for CV
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

print(f"\nCombined dataset: {len(X_combined)} rows, {y_combined.sum()} tickets")

# ============================================================
# STEP 3: MODEL TRAINING WITH CV
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Model Training with 10-Fold Stratified CV")
print("=" * 60)

# Calculate class weights
n_neg = (y_combined == 0).sum()
n_pos = (y_combined == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"Scale pos weight: {scale_pos_weight:.1f}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
models = {}
cv_proba = {}

# --- LightGBM ---
if HAS_LGBM:
    print("\nTraining LightGBM...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Get CV predictions
    lgbm_proba = cross_val_predict(lgbm_model, X_combined, y_combined, cv=cv, method='predict_proba')[:, 1]
    lgbm_pred = (lgbm_proba >= 0.5).astype(int)
    
    cv_f1 = f1_score(y_combined, lgbm_pred)
    cv_recall = recall_score(y_combined, lgbm_pred)
    cv_precision = precision_score(y_combined, lgbm_pred, zero_division=0)
    
    results['LightGBM'] = {'cv_f1': cv_f1, 'cv_recall': cv_recall, 'cv_precision': cv_precision}
    cv_proba['LightGBM'] = lgbm_proba
    
    # Fit on full combined data
    lgbm_model.fit(X_combined, y_combined)
    models['LightGBM'] = lgbm_model
    
    print(f"  CV F1: {cv_f1:.4f}, Recall: {cv_recall:.4f}, Precision: {cv_precision:.4f}")

# --- XGBoost ---
if HAS_XGB:
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    xgb_proba = cross_val_predict(xgb_model, X_combined, y_combined, cv=cv, method='predict_proba')[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    
    cv_f1 = f1_score(y_combined, xgb_pred)
    cv_recall = recall_score(y_combined, xgb_pred)
    cv_precision = precision_score(y_combined, xgb_pred, zero_division=0)
    
    results['XGBoost'] = {'cv_f1': cv_f1, 'cv_recall': cv_recall, 'cv_precision': cv_precision}
    cv_proba['XGBoost'] = xgb_proba
    
    xgb_model.fit(X_combined, y_combined)
    models['XGBoost'] = xgb_model
    
    print(f"  CV F1: {cv_f1:.4f}, Recall: {cv_recall:.4f}, Precision: {cv_precision:.4f}")

# --- Random Forest ---
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_proba = cross_val_predict(rf_model, X_combined, y_combined, cv=cv, method='predict_proba')[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)

cv_f1 = f1_score(y_combined, rf_pred)
cv_recall = recall_score(y_combined, rf_pred)
cv_precision = precision_score(y_combined, rf_pred, zero_division=0)

results['RandomForest'] = {'cv_f1': cv_f1, 'cv_recall': cv_recall, 'cv_precision': cv_precision}
cv_proba['RandomForest'] = rf_proba

rf_model.fit(X_combined, y_combined)
models['RandomForest'] = rf_model

print(f"  CV F1: {cv_f1:.4f}, Recall: {cv_recall:.4f}, Precision: {cv_precision:.4f}")

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=0.1
)

lr_proba = cross_val_predict(lr_model, X_combined_scaled, y_combined, cv=cv, method='predict_proba')[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)

cv_f1 = f1_score(y_combined, lr_pred)
cv_recall = recall_score(y_combined, lr_pred)
cv_precision = precision_score(y_combined, lr_pred, zero_division=0)

results['LogisticReg'] = {'cv_f1': cv_f1, 'cv_recall': cv_recall, 'cv_precision': cv_precision}
cv_proba['LogisticReg'] = lr_proba

lr_model.fit(X_combined_scaled, y_combined)
models['LogisticReg'] = (lr_model, scaler)

print(f"  CV F1: {cv_f1:.4f}, Recall: {cv_recall:.4f}, Precision: {cv_precision:.4f}")

# ============================================================
# STEP 4: THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Threshold Optimization")
print("=" * 60)

# Find best model by CV F1
best_model_name = max(results.keys(), key=lambda k: results[k]['cv_f1'])
print(f"Best model by CV F1: {best_model_name}")

# Get validation predictions from best model
if best_model_name == 'LogisticReg':
    X_val_scaled = scaler.transform(X_val)
    val_proba = models['LogisticReg'][0].predict_proba(X_val_scaled)[:, 1]
else:
    val_proba = models[best_model_name].predict_proba(X_val)[:, 1]

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[min(best_f1_idx, len(thresholds)-1)]
best_f1 = f1_scores[best_f1_idx]

print(f"\nBest F1 threshold: {best_f1_threshold:.4f}")
print(f"  F1: {best_f1:.4f}, Precision: {precisions[best_f1_idx]:.4f}, Recall: {recalls[best_f1_idx]:.4f}")

# Find threshold for recall >= 0.70 with best precision
recall_target = 0.70
valid_indices = np.where(recalls >= recall_target)[0]
if len(valid_indices) > 0:
    # Among valid indices, find one with best precision
    best_prec_idx = valid_indices[np.argmax(precisions[valid_indices])]
    recall_threshold = thresholds[min(best_prec_idx, len(thresholds)-1)]
    recall_prec = precisions[best_prec_idx]
    recall_rec = recalls[best_prec_idx]
    recall_f1 = 2 * recall_prec * recall_rec / (recall_prec + recall_rec + 1e-10)
else:
    # Use lowest threshold
    recall_threshold = 0.1
    recall_prec = precisions[-1]
    recall_rec = recalls[-1]
    recall_f1 = 2 * recall_prec * recall_rec / (recall_prec + recall_rec + 1e-10)

print(f"\nRecall-optimized threshold: {recall_threshold:.4f}")
print(f"  F1: {recall_f1:.4f}, Precision: {recall_prec:.4f}, Recall: {recall_rec:.4f}")

# ============================================================
# STEP 5: VALIDATION SET EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Validation Set Evaluation")
print("=" * 60)

for model_name in results.keys():
    if model_name == 'LogisticReg':
        X_val_scaled = scaler.transform(X_val)
        proba = models['LogisticReg'][0].predict_proba(X_val_scaled)[:, 1]
    else:
        proba = models[model_name].predict_proba(X_val)[:, 1]
    
    # Evaluate at 0.5 threshold
    pred = (proba >= 0.5).astype(int)
    val_f1 = f1_score(y_val, pred)
    val_recall = recall_score(y_val, pred)
    val_precision = precision_score(y_val, pred, zero_division=0)
    
    results[model_name]['val_f1'] = val_f1
    results[model_name]['val_recall'] = val_recall
    results[model_name]['val_precision'] = val_precision
    results[model_name]['val_flagged'] = pred.sum()

# ============================================================
# STEP 6: GENERATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Generate Submissions")
print("=" * 60)

# Use best model for test predictions
if best_model_name == 'LogisticReg':
    X_test_scaled = scaler.transform(X_test)
    test_proba = models['LogisticReg'][0].predict_proba(X_test_scaled)[:, 1]
else:
    test_proba = models[best_model_name].predict_proba(X_test)[:, 1]

# F1-optimized submission
test_pred_f1 = (test_proba >= best_f1_threshold).astype(int)
submission_f1 = pd.DataFrame({
    'call_id': test_features['call_id'],
    'has_ticket': test_pred_f1.astype(bool)
})
submission_f1.to_csv(f"{DATA_DIR}/submission_real_f1.csv", index=False)
print(f"F1-optimized: {test_pred_f1.sum()} flagged ({100*test_pred_f1.mean():.2f}%)")

# Recall-optimized submission
test_pred_recall = (test_proba >= recall_threshold).astype(int)
submission_recall = pd.DataFrame({
    'call_id': test_features['call_id'],
    'has_ticket': test_pred_recall.astype(bool)
})
submission_recall.to_csv(f"{DATA_DIR}/submission_real_recall.csv", index=False)
print(f"Recall-optimized: {test_pred_recall.sum()} flagged ({100*test_pred_recall.mean():.2f}%)")

# ============================================================
# STEP 7: FINAL REPORT
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: FINAL REPORT")
print("=" * 60)

print("\n" + "-" * 100)
print(f"{'Model':<15} | {'CV F1':>8} | {'CV Recall':>10} | {'CV Prec':>10} | {'Val F1':>8} | {'Val Recall':>10} | {'Flagged':>8}")
print("-" * 100)

for model_name, metrics in results.items():
    print(f"{model_name:<15} | {metrics['cv_f1']:>8.4f} | {metrics['cv_recall']:>10.4f} | {metrics['cv_precision']:>10.4f} | {metrics['val_f1']:>8.4f} | {metrics['val_recall']:>10.4f} | {metrics['val_flagged']:>8}")

print("-" * 100)

# Check targets
print("\n📊 TARGET CHECK:")
best_val_f1 = max(r['val_f1'] for r in results.values())
best_val_recall = max(r['val_recall'] for r in results.values())

target_f1 = 0.40
target_recall = 0.60

f1_status = "✅" if best_val_f1 > target_f1 else "❌"
recall_status = "✅" if best_val_recall > target_recall else "❌"

print(f"  {f1_status} Val F1 > {target_f1}: Best = {best_val_f1:.4f}")
print(f"  {recall_status} Val Recall > {target_recall}: Best = {best_val_recall:.4f}")

# Feature importance (if available)
if best_model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
    print("\n📈 TOP 10 FEATURE IMPORTANCES:")
    if best_model_name == 'RandomForest':
        importances = models[best_model_name].feature_importances_
    elif best_model_name == 'LightGBM':
        importances = models[best_model_name].feature_importances_
    else:
        importances = models[best_model_name].feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

print("\n" + "=" * 60)
print("SUBMISSIONS SAVED:")
print(f"  1. {DATA_DIR}/submission_real_f1.csv (F1-optimized)")
print(f"  2. {DATA_DIR}/submission_real_recall.csv (Recall-optimized)")
print("=" * 60)
