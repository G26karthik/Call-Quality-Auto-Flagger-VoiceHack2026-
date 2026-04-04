"""
PRODUCTION PIPELINE - REAL COMPETITION
Optimized for: HIGH RECALL + SPEED + ROBUSTNESS

Key features:
1. Auto-detects column name changes
2. Optimized for maximum ticket detection (high recall)
3. Fast execution (parallel processing)
4. Robust to new columns
5. Ensemble tuned for recall > precision
"""

import pandas as pd
import numpy as np
import json
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("=" * 80)
print("PRODUCTION PIPELINE - REAL COMPETITION")
print("Optimized for: Maximum Ticket Detection + Speed")
print("=" * 80)

start_time = time.time()

# ============================================================================
# COLUMN MAPPING - AUTO-DETECT RENAMES
# ============================================================================

def detect_column_mapping(df):
    """Auto-detect column names in case of renames"""
    col_map = {}
    
    # Define possible column name variations
    patterns = {
        'call_id': ['call_id', 'id', 'call_identifier'],
        'outcome': ['outcome', 'call_outcome', 'result'],
        'call_duration': ['call_duration', 'duration', 'call_length'],
        'whisper_mismatch_count': ['whisper_mismatch_count', 'mismatch_count'],
        'response_completeness': ['response_completeness', 'completeness'],
        'responses_json': ['responses_json', 'responses', 'answers_json'],
        'user_word_count': ['user_word_count', 'user_words'],
        'agent_word_count': ['agent_word_count', 'agent_words'],
    }
    
    cols_lower = {c.lower(): c for c in df.columns}
    
    for standard_name, variations in patterns.items():
        for var in variations:
            if var.lower() in cols_lower:
                col_map[standard_name] = cols_lower[var.lower()]
                break
    
    return col_map

# ============================================================================
# FEATURE ENGINEERING - FAST & ROBUST
# ============================================================================

def extract_features_fast(row, col_map):
    """Fast feature extraction with auto-column mapping"""
    
    # Use mapped column names or fallback to standard names
    def get_val(std_name, default=0):
        actual_name = col_map.get(std_name, std_name)
        return row.get(actual_name, default)
    
    features = {
        # Core signals
        'whisper_mismatch_count': int(get_val('whisper_mismatch_count', 0)),
        'has_whisper_mismatch': int(get_val('whisper_mismatch_count', 0) > 0),
        'response_completeness': float(get_val('response_completeness', 1.0)),
        'incomplete_responses': int(get_val('response_completeness', 1.0) < 1.0),
        
        # Outcome signals
        'outcome_wrong_number': int(str(get_val('outcome', '')).lower() == 'wrong_number'),
        'outcome_opted_out': int(str(get_val('outcome', '')).lower() == 'opted_out'),
        'outcome_escalated': int(str(get_val('outcome', '')).lower() == 'escalated'),
        'outcome_completed': int(str(get_val('outcome', '')).lower() == 'completed'),
        'outcome_incomplete': int(str(get_val('outcome', '')).lower() == 'incomplete'),
        
        # Engagement
        'user_word_count_val': get_val('user_word_count', 0),
        'low_engagement': int(get_val('user_word_count', 0) < 20),
        'high_engagement': int(get_val('user_word_count', 0) > 100),
    }
    
    # Ticket signal
    features['any_ticket_signal'] = int(
        features['has_whisper_mismatch'] or
        features['outcome_wrong_number'] or
        features['outcome_escalated'] or
        (features['outcome_completed'] and features['incomplete_responses'])
    )
    
    return features


def extract_response_features_fast(responses_json):
    """Fast response feature extraction"""
    result = {
        'answered_ratio': 0.0,
        'answered_count': 0,
        'has_empty_answers': 0,
        'total_questions': 0
    }
    
    if pd.isna(responses_json) or responses_json == '':
        return result
    
    try:
        responses = json.loads(responses_json)
        total = len(responses)
        answered = sum(1 for r in responses if str(r.get('answer', '')).strip() != '')
        
        result['total_questions'] = total
        result['answered_count'] = answered
        if total > 0:
            result['answered_ratio'] = answered / total
            result['has_empty_answers'] = int(answered < total)
    except:
        pass
    
    return result


def engineer_features_production(df, label_encoders=None, is_train=False, col_map=None):
    """Production feature engineering - fast and robust"""
    df = df.copy()
    
    if col_map is None:
        col_map = detect_column_mapping(df)
    
    print(f"  Processing {len(df)} rows...")
    
    # Extract features
    signal_features = df.apply(lambda row: extract_features_fast(row, col_map), axis=1)
    signal_df = pd.DataFrame(signal_features.tolist(), index=df.index)
    df = pd.concat([df, signal_df], axis=1)
    
    # Response features
    resp_col = col_map.get('responses_json', 'responses_json')
    if resp_col in df.columns:
        response_features = df[resp_col].apply(extract_response_features_fast)
        response_df = pd.DataFrame(response_features.tolist(), index=df.index)
        df = pd.concat([df, response_df], axis=1)
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Categorical encoding
    categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                       'patient_state', 'direction']
    
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
                le = label_encoders.get(col)
                if le:
                    df[col + '_enc'] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
    
    # Interaction features
    if 'whisper_mismatch_count' in df.columns and 'response_completeness' in df.columns:
        df['whisper_x_completeness'] = df['whisper_mismatch_count'] * (1 - df['response_completeness'])
    if 'outcome_enc' in df.columns and 'answered_count' in df.columns:
        df['outcome_x_answered'] = df['outcome_enc'] * df['answered_count']
    if 'any_ticket_signal' in df.columns and 'call_duration' in df.columns:
        df['ticket_signal_x_duration'] = df['any_ticket_signal'] * df['call_duration']
    if 'user_word_count_val' in df.columns and 'response_completeness' in df.columns:
        df['user_words_x_completeness'] = df['user_word_count_val'] * (1 - df['response_completeness'])
    
    # Drop leakage columns + timestamps
    drop_cols = [
        'call_id', 'patient_name_anon', 'attempted_at', 'scheduled_at',
        'organization_id', 'product_id',
        'ticket_has_reason', 'ticket_priority', 'ticket_status',
        'ticket_initial_notes', 'ticket_resolution_notes',
        'ticket_raised_at', 'ticket_resolved_at',
        'validation_notes', 'transcript_text', 'responses_json', 'whisper_transcript',
        'form_submitted', 'max_time_in_call'
    ]
    # Add any ticket_cat columns
    drop_cols += [c for c in df.columns if 'ticket_cat' in c.lower()]
    # Add any new timestamp columns
    drop_cols += [c for c in df.columns if 'timestamp' in c.lower() or '_at' in c.lower()]
    # Add categorical originals
    drop_cols += categorical_cols
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.fillna(0)
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df, label_encoders, col_map


# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

DATA_DIR = "../Datasets/csv"

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_csv(f"{DATA_DIR}/hackathon_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/hackathon_val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/hackathon_test.csv")

print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Detect column mapping
col_map = detect_column_mapping(train_df)
print(f"✓ Detected {len(col_map)} column mappings")

# Combine train + val
combined_df = pd.concat([train_df, val_df], ignore_index=True)
test_call_ids = test_df['call_id'].copy()

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

combined_processed, label_encs, col_map = engineer_features_production(
    combined_df, is_train=True, col_map=col_map
)
test_processed, _, _ = engineer_features_production(
    test_df, label_encoders=label_encs, is_train=False, col_map=col_map
)

X_combined = combined_processed.drop('has_ticket', axis=1)
y_combined = combined_processed['has_ticket'].astype(int)
X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Align features
common_cols = sorted(list(set(X_combined.columns) & set(X_test.columns)))
X_combined = X_combined[common_cols]
X_test = X_test[common_cols]

print(f"✓ Features: {len(common_cols)}")

# ============================================================================
# TRAIN MODELS - OPTIMIZED FOR RECALL
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MODELS (RECALL-OPTIMIZED)")
print("=" * 80)

pos_count = y_combined.sum()
neg_count = len(y_combined) - pos_count
scale_pos_weight = neg_count / pos_count

# Increase scale_pos_weight to favor recall
recall_boost = 1.5  # Boost positive class weight by 50%
scale_pos_weight_boosted = scale_pos_weight * recall_boost

# USING v5's PROVEN MODEL CONFIGURATION + RECALL OPTIMIZATION
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=150,  # v5 setting
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=15,
        reg_lambda=1.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight_boosted,  # BOOSTED for recall
        random_state=42,
        verbose=-1,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=150,  # v5 setting
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight_boosted,  # BOOSTED for recall
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,  # v5 setting
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        C=0.2,
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )
}

# Train all models (parallel where possible)
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_combined, y_combined)

print(f"✓ Trained {len(models)} models")

# ============================================================================
# PREDICT - USING v5's ENSEMBLE APPROACH + RECALL OPTIMIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS (v5 ENSEMBLE + RECALL-OPTIMIZED)")
print("=" * 80)

# Get probabilities from all models
test_probas = []
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    test_probas.append(proba)
    print(f"  {name}: mean proba = {proba.mean():.4f}")

# Use v5's proven ensemble weights (from Optuna optimization)
# These weights were found optimal for F1 score
ensemble_weights = {
    'LightGBM': 0.434,
    'XGBoost': 0.099,
    'Random Forest': 0.207,
    'Logistic Regression': 0.260
}

print(f"\n✓ Using v5's optimized ensemble weights:")
for name, weight in ensemble_weights.items():
    print(f"    {name:20s}: {weight:.3f}")

# Weighted ensemble
ensemble_proba = np.zeros(len(X_test))
for (name, weight), proba in zip(ensemble_weights.items(), test_probas):
    ensemble_proba += weight * proba

# RECALL-OPTIMIZED THRESHOLD (lower than v5's 0.68)
# v5 used 0.68 for balanced F1
# We use 0.38 for maximum recall (catch more tickets)
threshold = 0.38

ensemble_pred = (ensemble_proba >= threshold).astype(int)

tickets_flagged = ensemble_pred.sum()
print(f"✓ Tickets flagged: {tickets_flagged} / {len(test_df)} ({100*tickets_flagged/len(test_df):.1f}%)")
print(f"✓ Threshold: {threshold} (lowered for recall)")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': ensemble_pred
})

submission.to_csv('submission_production.csv', index=False)

elapsed = time.time() - start_time
print(f"\n✓ Submission saved: submission_production.csv")
print(f"✓ Total time: {elapsed:.1f} seconds")

print("\n" + "=" * 80)
print("PRODUCTION STATS")
print("=" * 80)
print(f"  Models:           4 (v5 ensemble)")
print(f"  Ensemble:         v5 optimized weights")
print(f"  Speed:            {elapsed:.1f}s")
print(f"  Tickets flagged:  {tickets_flagged}")
print(f"  Threshold:        {threshold} (recall-optimized, v5 used 0.68)")
print(f"  Positive boost:   {recall_boost}x")
print(f"  Auto-mapping:     {len(col_map)} columns")
print("=" * 80)
