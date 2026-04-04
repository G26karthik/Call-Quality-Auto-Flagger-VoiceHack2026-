"""
CareCaller Hackathon - ML Pipeline v4b
Balance: Keep some domain patterns but with regularization + CV
- Keep generalizable patterns (whisper, outcome-based, response completeness)
- Add back SOME text patterns but broader ones
- 5-fold StratifiedKFold CV
- Moderate regularization
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, precision_recall_curve
from lightgbm import LGBMClassifier

print("=" * 70)
print("ML PIPELINE v4b - Balanced Anti-Overfitting")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
DATA_DIR = "../Datasets/csv"
train_df = pd.read_csv(f"{DATA_DIR}/hackathon_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/hackathon_val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/hackathon_test.csv")

# Combine train + val for CV
combined_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Combined: {len(combined_df)}, Test: {len(test_df)}")

test_call_ids = test_df['call_id'].copy()

pos_count = combined_df['has_ticket'].sum()
neg_count = len(combined_df) - pos_count
scale_pos_weight = neg_count / pos_count

# ============================================================================
# BALANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("BALANCED FEATURE ENGINEERING")
print("=" * 70)

categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                    'patient_state', 'direction']

def extract_balanced_features(row):
    """Extract features with balance between domain knowledge and generalization"""
    notes = str(row.get('validation_notes', '')).lower()
    outcome = str(row.get('outcome', '')).lower()
    
    features = {}
    
    # === STRUCTURAL FEATURES (always generalizable) ===
    features['whisper_mismatch_count'] = int(row.get('whisper_mismatch_count', 0))
    features['has_whisper_mismatch'] = int(features['whisper_mismatch_count'] > 0)
    features['response_completeness'] = float(row.get('response_completeness', 1.0))
    features['incomplete_responses'] = int(features['response_completeness'] < 1.0)
    
    # === OUTCOME-BASED (generalizable) ===
    features['outcome_wrong_number'] = int(outcome == 'wrong_number')
    features['outcome_opted_out'] = int(outcome == 'opted_out')
    features['outcome_escalated'] = int(outcome == 'escalated')
    features['outcome_completed'] = int(outcome == 'completed')
    features['outcome_incomplete'] = int(outcome == 'incomplete')
    
    # === BROADER TEXT PATTERNS (more generalizable than exact phrases) ===
    # Instead of "dosage guidance", look for any guidance/advice mention
    features['mentions_guidance'] = int('guidance' in notes or 'advice' in notes or 'recommend' in notes)
    
    # Instead of exact "miscategorization", look for any categorization issue
    features['mentions_categorization'] = int('categor' in notes or 'classif' in notes)
    
    # Error/issue indicators (broad)
    features['mentions_error'] = int('error' in notes or 'issue' in notes or 'problem' in notes)
    
    # Mismatch/discrepancy (broad)
    features['mentions_mismatch'] = int('mismatch' in notes or 'discrepancy' in notes or 'differ' in notes)
    
    # Correction indicator
    features['mentions_correction'] = int('correct' in notes and ('outcome' in notes or 'classif' in notes))
    
    # === COMBINED SIGNALS ===
    features['any_structural_issue'] = int(
        features['has_whisper_mismatch'] or
        features['outcome_wrong_number'] or
        features['outcome_escalated'] or
        (features['outcome_completed'] and features['incomplete_responses'])
    )
    
    features['any_text_issue'] = int(
        features['mentions_guidance'] or
        features['mentions_categorization'] or
        features['mentions_correction']
    )
    
    features['any_ticket_signal'] = int(
        features['any_structural_issue'] or
        features['any_text_issue']
    )
    
    return features


def extract_response_features(responses_json):
    """Extract response features"""
    result = {
        'answered_ratio': 0.0,
        'empty_count': 0,
        'total_questions': 0
    }
    
    if pd.isna(responses_json) or responses_json == '':
        return result
    
    try:
        responses = json.loads(responses_json)
        total = len(responses)
        answered = sum(1 for r in responses if str(r.get('answer', '')).strip() != '')
        
        result['total_questions'] = total
        result['empty_count'] = total - answered
        if total > 0:
            result['answered_ratio'] = answered / total
    except:
        pass
    
    return result


def engineer_features_v4b(df, label_encoders=None, is_train=False):
    """Balanced feature engineering"""
    df = df.copy()
    
    # Extract features
    signal_features = df.apply(extract_balanced_features, axis=1)
    signal_df = pd.DataFrame(signal_features.tolist(), index=df.index)
    df = pd.concat([df, signal_df], axis=1)
    
    response_features = df['responses_json'].apply(extract_response_features)
    response_df = pd.DataFrame(response_features.tolist(), index=df.index)
    df = pd.concat([df, response_df], axis=1)
    
    # Label encode
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('unknown').astype(str)
                df[col + '_enc'] = le.fit_transform(df[col])
                label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
                le = label_encoders[col]
                df[col + '_enc'] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # Drop columns
    drop_cols = [
        'call_id', 'patient_name_anon', 'attempted_at', 'scheduled_at',
        'organization_id', 'product_id',
        'ticket_has_reason', 'ticket_priority', 'ticket_status',
        'ticket_initial_notes', 'ticket_resolution_notes',
        'ticket_cat_audio_issue', 'ticket_cat_audio_notes',
        'ticket_cat_elevenlabs', 'ticket_cat_elevenlabs_notes',
        'ticket_cat_openai', 'ticket_cat_openai_notes',
        'ticket_cat_supabase', 'ticket_cat_supabase_notes',
        'ticket_cat_scheduler_aws', 'ticket_cat_scheduler_aws_notes',
        'ticket_cat_other', 'ticket_cat_other_notes',
        'ticket_raised_at', 'ticket_resolved_at',
        'transcript_text', 'validation_notes', 'responses_json', 'whisper_transcript',
        'form_submitted', 'max_time_in_call'
    ] + categorical_cols
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.fillna(0)
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df, label_encoders


# Process data
print("\nProcessing combined...")
combined_processed, label_encs = engineer_features_v4b(combined_df, is_train=True)

print("Processing test...")
test_processed, _ = engineer_features_v4b(test_df, label_encoders=label_encs, is_train=False)

# Prepare X, y
X_combined = combined_processed.drop('has_ticket', axis=1)
y_combined = combined_processed['has_ticket'].astype(int)

X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Align columns
common_cols = sorted(list(set(X_combined.columns) & set(X_test.columns)))
X_combined = X_combined[common_cols]
X_test = X_test[common_cols]

print(f"\nFeature count: {len(common_cols)}")

# ============================================================================
# 5-FOLD CV WITH MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CV")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LightGBM': LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=12,
        min_child_samples=15,
        reg_lambda=0.5,
        reg_alpha=0.2,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=15,
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=0.5,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
}

cv_probas = {}

for name, model in models.items():
    y_cv_proba = cross_val_predict(model, X_combined, y_combined, cv=cv, method='predict_proba')[:, 1]
    cv_probas[name] = y_cv_proba
    
    # Optimal threshold for this model
    precisions, recalls, thresholds = precision_recall_curve(y_combined, y_cv_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"{name}: CV F1={best_f1:.4f} (threshold={best_thresh:.3f})")
    
    # Fit on full data
    model.fit(X_combined, y_combined)

# ============================================================================
# ENSEMBLE WITH THRESHOLD TUNING
# ============================================================================
print("\n" + "=" * 70)
print("ENSEMBLE")
print("=" * 70)

# Weighted average (give more weight to LightGBM)
weights = {'LightGBM': 0.5, 'Random Forest': 0.3, 'Logistic Regression': 0.2}
ensemble_cv_proba = sum(cv_probas[name] * weights[name] for name in models)

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_combined, ensemble_cv_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
best_cv_f1 = f1_scores[best_idx]

print(f"Ensemble optimal threshold: {best_threshold:.4f}")
print(f"Ensemble best CV F1: {best_cv_f1:.4f}")

ensemble_cv_pred = (ensemble_cv_proba >= best_threshold).astype(int)
print(f"Ensemble CV - P: {precision_score(y_combined, ensemble_cv_pred):.4f}, R: {recall_score(y_combined, ensemble_cv_pred):.4f}")

# ============================================================================
# EVALUATE ON ORIGINAL VAL SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SET (Original Split)")
print("=" * 70)

val_indices = range(len(train_df), len(combined_df))
y_val_true = y_combined.iloc[val_indices].values
ensemble_val_proba = ensemble_cv_proba[val_indices]
ensemble_val_pred = (ensemble_val_proba >= best_threshold).astype(int)

val_f1 = f1_score(y_val_true, ensemble_val_pred)
val_precision = precision_score(y_val_true, ensemble_val_pred)
val_recall = recall_score(y_val_true, ensemble_val_pred)

print(f"Val F1: {val_f1:.4f}")
print(f"Val Precision: {val_precision:.4f}")
print(f"Val Recall: {val_recall:.4f}")

print("\n" + classification_report(y_val_true, ensemble_val_pred, target_names=['No Ticket', 'Has Ticket']))

# ============================================================================
# TEST PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

test_probas = []
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    test_probas.append(proba * weights[name])

ensemble_test_proba = sum(test_probas)
ensemble_test_pred = (ensemble_test_proba >= best_threshold).astype(int)

print(f"Test tickets flagged: {ensemble_test_pred.sum()}")

# Save submission
submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': ensemble_test_pred
})
submission.to_csv('submission_ml_v4.csv', index=False)
print(f"Saved: submission_ml_v4.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
+---------------------------+----------+----------+------------+
| Method                    | CV F1    | Val F1   | Test Flags |
+---------------------------+----------+----------+------------+
| Rule-based                |   N/A    |  1.00    |     18     |
| ML v3 (Overfit)           |   N/A    |  0.95*   |     15     |
| ML v4 Ensemble            |  {best_cv_f1:.2f}    |  {val_f1:.2f}    |     {ensemble_test_pred.sum()}     |
+---------------------------+----------+----------+------------+
* v3 Val F1=0.95 but Private LB dropped to 0.85 (overfit)

Changes in v4:
- Removed hyper-specific patterns (exact phrase matches)
- Used broader text patterns (guidance/advice, categor/classif)
- 5-fold CV instead of single split
- Weighted ensemble (LGBM 50%, RF 30%, LR 20%)
- Moderate regularization

Target: Val F1 > 0.90 with better generalization
Achieved: Val F1 = {val_f1:.4f}
""")

# Feature importance
print("\nTop 15 features (LightGBM):")
importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance': models['LightGBM'].feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(15).to_string(index=False))
