"""
CareCaller Hackathon - ML Pipeline v4
Focus: Reduce overfitting while keeping domain knowledge
- Fewer binary flags (only whisper_mismatch, any_ticket_signal, response_completeness)
- 5-fold StratifiedKFold CV on train+val combined
- Lower model complexity (max_depth=3-4, higher regularization)
- Simple ensemble (LR + LightGBM + RF averaging)
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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from lightgbm import LGBMClassifier

print("=" * 70)
print("ML PIPELINE v4 - Anti-Overfitting")
print("Focus: Generalization over train fit")
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
print(f"Combined train+val: {len(combined_df)}")
print(f"Test: {len(test_df)}")

test_call_ids = test_df['call_id'].copy()

pos_count = combined_df['has_ticket'].sum()
neg_count = len(combined_df) - pos_count
scale_pos_weight = neg_count / pos_count
print(f"Class imbalance: {scale_pos_weight:.2f}")

# ============================================================================
# SIMPLIFIED FEATURE ENGINEERING (Less Overfitting)
# ============================================================================
print("\n" + "=" * 70)
print("SIMPLIFIED FEATURE ENGINEERING")
print("=" * 70)

categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                    'patient_state', 'direction']

def extract_simple_features(row):
    """Extract ONLY generalizable domain features - no specific text patterns"""
    notes = str(row.get('validation_notes', '')).lower()
    
    # ONLY keep highly generalizable signals
    features = {
        # Whisper mismatch - structural, not text-based
        'whisper_mismatch_count': int(row.get('whisper_mismatch_count', 0)),
        'has_whisper_mismatch': int(row.get('whisper_mismatch_count', 0) > 0),
        
        # Response completeness - structural
        'response_completeness': float(row.get('response_completeness', 1.0)),
        'incomplete_responses': int(row.get('response_completeness', 1.0) < 1.0),
        
        # Outcome-based (generalizable across datasets)
        'outcome_wrong_number': int(str(row.get('outcome', '')).lower() == 'wrong_number'),
        'outcome_opted_out': int(str(row.get('outcome', '')).lower() == 'opted_out'),
        'outcome_escalated': int(str(row.get('outcome', '')).lower() == 'escalated'),
        'outcome_completed': int(str(row.get('outcome', '')).lower() == 'completed'),
        
        # General error indicator (less specific than individual patterns)
        'has_any_error_word': int(any(w in notes for w in ['error', 'issue', 'incorrect', 'mismatch'])),
        
        # Engagement metrics (structural, generalizable)
        'low_engagement': int(row.get('user_word_count', 0) < 20),
        'high_engagement': int(row.get('user_word_count', 0) > 100),
    }
    
    # Combined signal - simplified
    features['any_ticket_signal'] = int(
        features['has_whisper_mismatch'] or
        features['outcome_wrong_number'] or
        features['outcome_escalated'] or
        (features['outcome_completed'] and features['incomplete_responses'])
    )
    
    return features


def extract_response_features(responses_json):
    """Simple response features"""
    result = {
        'answered_ratio': 0.0,
        'has_empty_answers': 0
    }
    
    if pd.isna(responses_json) or responses_json == '':
        return result
    
    try:
        responses = json.loads(responses_json)
        total = len(responses)
        answered = sum(1 for r in responses if str(r.get('answer', '')).strip() != '')
        
        if total > 0:
            result['answered_ratio'] = answered / total
            result['has_empty_answers'] = int(answered < total)
    except:
        pass
    
    return result


def engineer_features_v4(df, label_encoders=None, scaler=None, is_train=False):
    """Simplified feature engineering for better generalization"""
    df = df.copy()
    
    # Extract simplified features
    print("  Extracting simplified domain features...")
    signal_features = df.apply(extract_simple_features, axis=1)
    signal_df = pd.DataFrame(signal_features.tolist(), index=df.index)
    df = pd.concat([df, signal_df], axis=1)
    
    # Response features
    print("  Extracting response features...")
    response_features = df['responses_json'].apply(extract_response_features)
    response_df = pd.DataFrame(response_features.tolist(), index=df.index)
    df = pd.concat([df, response_df], axis=1)
    
    # Keep only essential numeric columns
    numeric_cols = [
        'call_duration', 'attempt_number', 'question_count', 
        'turn_count', 'user_turn_count', 'agent_turn_count',
        'user_word_count', 'agent_word_count', 
        'avg_user_turn_words', 'avg_agent_turn_words',
        'interruption_count', 'hour_of_day'
    ]
    
    # Label encode categoricals
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
    
    # Drop unwanted columns
    drop_cols = [
        'call_id', 'patient_name_anon', 'attempted_at', 'scheduled_at',
        'organization_id', 'product_id',  # Remove org/product to avoid overfitting
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
    
    # Remove duplicates
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df, label_encoders


# Process combined data
print("\nProcessing combined train+val...")
combined_processed, label_encs = engineer_features_v4(combined_df, is_train=True)

print("Processing test data...")
test_processed, _ = engineer_features_v4(test_df, label_encoders=label_encs, is_train=False)

# Prepare X, y
X_combined = combined_processed.drop('has_ticket', axis=1)
y_combined = combined_processed['has_ticket'].astype(int)

X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Align columns
common_cols = sorted(list(set(X_combined.columns) & set(X_test.columns)))
X_combined = X_combined[common_cols]
X_test = X_test[common_cols]

print(f"\nFeature count: {len(common_cols)}")
print(f"Features: {list(X_combined.columns)}")

# ============================================================================
# 5-FOLD STRATIFIED CV
# ============================================================================
print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models with LOWER complexity for generalization
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=3,  # Reduced from 5
        learning_rate=0.05,
        num_leaves=8,  # Reduced
        min_child_samples=20,  # Increased for regularization
        reg_lambda=1.0,  # L2 regularization
        reg_alpha=0.5,  # L1 regularization
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,  # Low depth
        min_samples_leaf=10,  # Regularization
        min_samples_split=20,
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=0.1,  # Strong regularization
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
}

cv_results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Get CV predictions
    y_cv_pred = cross_val_predict(model, X_combined, y_combined, cv=cv, method='predict')
    y_cv_proba = cross_val_predict(model, X_combined, y_combined, cv=cv, method='predict_proba')[:, 1]
    
    # Calculate CV metrics
    cv_f1 = f1_score(y_combined, y_cv_pred)
    cv_precision = precision_score(y_combined, y_cv_pred)
    cv_recall = recall_score(y_combined, y_cv_pred)
    
    cv_results[name] = {
        'model': model,
        'cv_f1': cv_f1,
        'cv_precision': cv_precision,
        'cv_recall': cv_recall,
        'cv_proba': y_cv_proba
    }
    
    print(f"  CV F1: {cv_f1:.4f}, Precision: {cv_precision:.4f}, Recall: {cv_recall:.4f}")
    
    # Fit on full data for test predictions
    model.fit(X_combined, y_combined)

# ============================================================================
# SIMPLE ENSEMBLE (Averaging)
# ============================================================================
print("\n" + "=" * 70)
print("SIMPLE ENSEMBLE (Average Probabilities)")
print("=" * 70)

# Average CV probabilities
ensemble_cv_proba = np.mean([cv_results[name]['cv_proba'] for name in models], axis=0)

# Find optimal threshold on CV
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred = (ensemble_cv_proba >= thresh).astype(int)
    f1 = f1_score(y_combined, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Ensemble CV F1: {best_f1:.4f}")

# Ensemble CV predictions
ensemble_cv_pred = (ensemble_cv_proba >= best_threshold).astype(int)
ensemble_cv_f1 = f1_score(y_combined, ensemble_cv_pred)
ensemble_cv_precision = precision_score(y_combined, ensemble_cv_pred)
ensemble_cv_recall = recall_score(y_combined, ensemble_cv_pred)

print(f"\nEnsemble CV Results:")
print(f"  F1: {ensemble_cv_f1:.4f}")
print(f"  Precision: {ensemble_cv_precision:.4f}")
print(f"  Recall: {ensemble_cv_recall:.4f}")

# ============================================================================
# EVALUATE ON ORIGINAL VAL SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SET EVALUATION (Original Split)")
print("=" * 70)

# Get val indices (last 144 rows of combined)
val_indices = range(len(train_df), len(combined_df))
y_val_true = y_combined.iloc[val_indices]
ensemble_val_proba = ensemble_cv_proba[val_indices]
ensemble_val_pred = (ensemble_val_proba >= best_threshold).astype(int)

val_f1 = f1_score(y_val_true, ensemble_val_pred)
val_precision = precision_score(y_val_true, ensemble_val_pred)
val_recall = recall_score(y_val_true, ensemble_val_pred)

print(f"Val F1: {val_f1:.4f}")
print(f"Val Precision: {val_precision:.4f}")
print(f"Val Recall: {val_recall:.4f}")

print("\nClassification Report (Val):")
print(classification_report(y_val_true, ensemble_val_pred, target_names=['No Ticket', 'Has Ticket']))

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

# Get test probabilities from each model
test_probas = []
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    test_probas.append(proba)
    print(f"{name}: mean proba = {proba.mean():.4f}")

# Average ensemble
ensemble_test_proba = np.mean(test_probas, axis=0)
ensemble_test_pred = (ensemble_test_proba >= best_threshold).astype(int)

print(f"\nEnsemble test predictions: {ensemble_test_pred.sum()} tickets flagged")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': ensemble_test_pred
})

submission.to_csv('submission_ml_v4.csv', index=False)
print(f"\nSaved: submission_ml_v4.csv")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(f"""
+---------------------------+----------+----------+------------+
| Method                    | CV F1    | Val F1   | Test Flags |
+---------------------------+----------+----------+------------+
| Rule-based                |   N/A    |  1.00    |     18     |
| ML v3 (Overfit)           |   N/A    |  0.95    |     15     |
| ML v4 Ensemble (This)     |  {ensemble_cv_f1:.2f}    |  {val_f1:.2f}    |     {ensemble_test_pred.sum()}     |
+---------------------------+----------+----------+------------+

Key changes in v4:
- Removed specific text patterns (dosage_guidance, miscategor, etc.)
- Kept only structural features (whisper_mismatch, response_completeness)
- Used 5-fold StratifiedKFold CV instead of single split
- Reduced model complexity (max_depth=3-4, regularization)
- Simple ensemble averaging (LR + LGBM + RF)

Target: Val F1 > 0.90 with better generalization
Achieved: Val F1 = {val_f1:.4f}
""")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (LightGBM)")
print("=" * 70)

lgbm_model = models['LightGBM']
importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))
