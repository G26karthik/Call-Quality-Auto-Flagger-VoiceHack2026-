"""
CareCaller Hackathon - ML Pipeline v5 CLEAN (NO DATA LEAKAGE)
CRITICAL FIX: Removed validation_notes usage - this was causing data leakage

Changes from v5:
- REMOVED: validation_notes usage in extract_simple_features
- REMOVED: has_any_error_word feature (was derived from validation_notes)
- All other improvements remain: full data, 10-fold CV, XGBoost, interactions, Optuna
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

print("=" * 80)
print("ML PIPELINE v5 CLEAN - NO DATA LEAKAGE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
DATA_DIR = "../Datasets/csv"
train_df = pd.read_csv(f"{DATA_DIR}/hackathon_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/hackathon_val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/hackathon_test.csv")

combined_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"\n✓ Using FULL dataset: {len(combined_df)} rows (train: {len(train_df)}, val: {len(val_df)})")
print(f"  Test: {len(test_df)} rows")

test_call_ids = test_df['call_id'].copy()

pos_count = combined_df['has_ticket'].sum()
neg_count = len(combined_df) - pos_count
scale_pos_weight = neg_count / pos_count
print(f"  Class imbalance: {pos_count} positive, {neg_count} negative (ratio: {scale_pos_weight:.2f})")

# ============================================================================
# FEATURE ENGINEERING - NO LEAKAGE
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING (CLEAN - NO validation_notes)")
print("=" * 80)

categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                    'patient_state', 'direction']

def extract_simple_features_clean(row):
    """
    Extract domain features WITHOUT using validation_notes
    CRITICAL: validation_notes contains human labels and causes data leakage
    """
    features = {
        # Whisper mismatch - structural signal
        'whisper_mismatch_count': int(row.get('whisper_mismatch_count', 0)),
        'has_whisper_mismatch': int(row.get('whisper_mismatch_count', 0) > 0),
        
        # Response completeness - structural signal
        'response_completeness': float(row.get('response_completeness', 1.0)),
        'incomplete_responses': int(row.get('response_completeness', 1.0) < 1.0),
        
        # Outcome-based - categorical signal
        'outcome_wrong_number': int(str(row.get('outcome', '')).lower() == 'wrong_number'),
        'outcome_opted_out': int(str(row.get('outcome', '')).lower() == 'opted_out'),
        'outcome_escalated': int(str(row.get('outcome', '')).lower() == 'escalated'),
        'outcome_completed': int(str(row.get('outcome', '')).lower() == 'completed'),
        'outcome_incomplete': int(str(row.get('outcome', '')).lower() == 'incomplete'),
        
        # REMOVED: has_any_error_word (was based on validation_notes - DATA LEAKAGE)
        
        # Engagement metrics
        'user_word_count_val': row.get('user_word_count', 0),
        'low_engagement': int(row.get('user_word_count', 0) < 20),
        'high_engagement': int(row.get('user_word_count', 0) > 100),
    }
    
    # Combined ticket signal
    features['any_ticket_signal'] = int(
        features['has_whisper_mismatch'] or
        features['outcome_wrong_number'] or
        features['outcome_escalated'] or
        (features['outcome_completed'] and features['incomplete_responses'])
    )
    
    return features


def extract_response_features(responses_json):
    """Response features from responses_json"""
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


def add_interaction_features(df):
    """Add interaction features"""
    interaction_count = 0
    
    # Interaction 1: whisper_mismatch_count × response_completeness
    if 'whisper_mismatch_count' in df.columns and 'response_completeness' in df.columns:
        df.loc[:, 'whisper_x_completeness'] = (
            df['whisper_mismatch_count'] * (1 - df['response_completeness'])
        )
        interaction_count += 1
    
    # Interaction 2: outcome_encoded × answered_count
    if 'outcome_enc' in df.columns and 'answered_count' in df.columns:
        df.loc[:, 'outcome_x_answered'] = df['outcome_enc'] * df['answered_count']
        interaction_count += 1
    
    # Interaction 3: any_ticket_signal × call_duration
    if 'any_ticket_signal' in df.columns and 'call_duration' in df.columns:
        df.loc[:, 'ticket_signal_x_duration'] = df['any_ticket_signal'] * df['call_duration']
        interaction_count += 1
    
    # Interaction 4: incomplete_responses × outcome_completed
    if 'incomplete_responses' in df.columns and 'outcome_completed' in df.columns:
        df.loc[:, 'incomplete_x_completed'] = df['incomplete_responses'] * df['outcome_completed']
        interaction_count += 1
    
    # Interaction 5: user_word_count × response_completeness
    if 'user_word_count_val' in df.columns and 'response_completeness' in df.columns:
        df.loc[:, 'user_words_x_completeness'] = (
            df['user_word_count_val'] * (1 - df['response_completeness'])
        )
        interaction_count += 1
    
    print(f"  Added {interaction_count} interaction features")
    
    return df


def engineer_features_clean(df, label_encoders=None, is_train=False):
    """Clean feature engineering WITHOUT validation_notes"""
    df = df.copy()
    
    # Extract clean features (NO validation_notes)
    print("  Extracting domain features (NO validation_notes)...")
    signal_features = df.apply(extract_simple_features_clean, axis=1)
    signal_df = pd.DataFrame(signal_features.tolist(), index=df.index)
    df = pd.concat([df, signal_df], axis=1)
    
    # Response features
    print("  Extracting response features...")
    response_features = df['responses_json'].apply(extract_response_features)
    response_df = pd.DataFrame(response_features.tolist(), index=df.index)
    df = pd.concat([df, response_df], axis=1)
    
    # Remove duplicate columns before proceeding
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Numeric columns
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
    
    # Add interaction features AFTER encoding
    print("  Adding interaction features...")
    df = add_interaction_features(df)
    
    # Drop unwanted columns (including ALL leakage columns)
    drop_cols = [
        'call_id', 'patient_name_anon', 'attempted_at', 'scheduled_at',
        'organization_id', 'product_id',
        # CRITICAL: Drop ALL ticket-related columns (data leakage)
        'ticket_has_reason', 'ticket_priority', 'ticket_status',
        'ticket_initial_notes', 'ticket_resolution_notes',
        'ticket_cat_audio_issue', 'ticket_cat_audio_notes',
        'ticket_cat_elevenlabs', 'ticket_cat_elevenlabs_notes',
        'ticket_cat_openai', 'ticket_cat_openai_notes',
        'ticket_cat_supabase', 'ticket_cat_supabase_notes',
        'ticket_cat_scheduler_aws', 'ticket_cat_scheduler_aws_notes',
        'ticket_cat_other', 'ticket_cat_other_notes',
        'ticket_raised_at', 'ticket_resolved_at',
        # CRITICAL: Drop validation_notes (data leakage)
        'validation_notes',
        'transcript_text', 'responses_json', 'whisper_transcript',
        'form_submitted', 'max_time_in_call'
    ] + categorical_cols
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.fillna(0)
    
    # Remove duplicates
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df, label_encoders


# Process data
print("\nProcessing combined data...")
combined_processed, label_encs = engineer_features_clean(combined_df, is_train=True)

print("Processing test data...")
test_processed, _ = engineer_features_clean(test_df, label_encoders=label_encs, is_train=False)

# Prepare X, y
X_combined = combined_processed.drop('has_ticket', axis=1)
y_combined = combined_processed['has_ticket'].astype(int)

X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Align columns
common_cols = sorted(list(set(X_combined.columns) & set(X_test.columns)))
X_combined = X_combined[common_cols]
X_test = X_test[common_cols]

print(f"\n✓ Feature count: {len(common_cols)}")
print(f"  Features: {list(X_combined.columns)}")

# ============================================================================
# 10-FOLD STRATIFIED CV
# ============================================================================
print("\n" + "=" * 80)
print("10-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'LightGBM': LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=15,
        reg_lambda=1.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
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
        max_iter=2000,  # Increased to avoid convergence warnings
        random_state=42
    )
}

print(f"\n✓ Training {len(models)} models with 10-fold CV...")

cv_results = {}
cv_scores_per_model = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Get CV predictions
    y_cv_pred = cross_val_predict(model, X_combined, y_combined, cv=cv, method='predict', n_jobs=-1)
    y_cv_proba = cross_val_predict(model, X_combined, y_combined, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Calculate per-fold scores
    fold_scores = []
    for train_idx, val_idx in cv.split(X_combined, y_combined):
        X_train_fold, X_val_fold = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
        y_train_fold, y_val_fold = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
        
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train_fold, y_train_fold)
        y_val_pred = model_clone.predict(X_val_fold)
        fold_f1 = f1_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_f1)
    
    cv_f1 = f1_score(y_combined, y_cv_pred)
    cv_precision = precision_score(y_combined, y_cv_pred)
    cv_recall = recall_score(y_combined, y_cv_pred)
    cv_f1_std = np.std(fold_scores)
    
    cv_results[name] = {
        'model': model,
        'cv_f1': cv_f1,
        'cv_f1_std': cv_f1_std,
        'cv_precision': cv_precision,
        'cv_recall': cv_recall,
        'cv_proba': y_cv_proba
    }
    
    print(f"  CV F1: {cv_f1:.4f} ± {cv_f1_std:.4f}")
    print(f"  Precision: {cv_precision:.4f}, Recall: {cv_recall:.4f}")
    
    # Fit on full data
    model.fit(X_combined, y_combined)

# ============================================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("OPTIMIZING ENSEMBLE WEIGHTS WITH OPTUNA")
print("=" * 80)

if OPTUNA_AVAILABLE:
    model_names = list(cv_results.keys())
    cv_probas = [cv_results[name]['cv_proba'] for name in model_names]
    
    def objective(trial):
        weights = [trial.suggest_float(f'weight_{name}', 0.0, 1.0) for name in model_names]
        weight_sum = sum(weights)
        if weight_sum == 0:
            return 0.0
        weights = [w / weight_sum for w in weights]
        
        weighted_proba = np.zeros(len(y_combined))
        for w, proba in zip(weights, cv_probas):
            weighted_proba += w * proba
        
        best_f1 = 0
        for thresh in np.arange(0.3, 0.7, 0.02):
            y_pred = (weighted_proba >= thresh).astype(int)
            f1 = f1_score(y_combined, y_pred)
            if f1 > best_f1:
                best_f1 = f1
        
        return best_f1
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    
    best_weights = [study.best_params[f'weight_{name}'] for name in model_names]
    weight_sum = sum(best_weights)
    best_weights = [w / weight_sum for w in best_weights]
    
    print(f"✓ Optimization complete")
    print(f"  Best CV F1: {study.best_value:.4f}")
    print(f"\n  Optimized weights:")
    for name, weight in zip(model_names, best_weights):
        print(f"    {name:20s}: {weight:.3f}")
    
    ensemble_cv_proba = np.zeros(len(y_combined))
    for weight, proba in zip(best_weights, cv_probas):
        ensemble_cv_proba += weight * proba

else:
    print("⚠ Optuna not available, using equal weights")
    best_weights = [1.0 / len(models)] * len(models)
    model_names = list(cv_results.keys())
    cv_probas = [cv_results[name]['cv_proba'] for name in model_names]
    ensemble_cv_proba = np.mean(cv_probas, axis=0)

# Find optimal threshold
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred = (ensemble_cv_proba >= thresh).astype(int)
    f1 = f1_score(y_combined, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n✓ Optimal threshold: {best_threshold:.3f}")

ensemble_cv_pred = (ensemble_cv_proba >= best_threshold).astype(int)
ensemble_cv_f1 = f1_score(y_combined, ensemble_cv_pred)
ensemble_cv_precision = precision_score(y_combined, ensemble_cv_pred)
ensemble_cv_recall = recall_score(y_combined, ensemble_cv_pred)

ensemble_fold_scores = []
for train_idx, val_idx in cv.split(X_combined, y_combined):
    val_proba = ensemble_cv_proba[val_idx]
    y_val_true = y_combined.iloc[val_idx]
    y_val_pred = (val_proba >= best_threshold).astype(int)
    fold_f1 = f1_score(y_val_true, y_val_pred)
    ensemble_fold_scores.append(fold_f1)

ensemble_cv_f1_std = np.std(ensemble_fold_scores)

print(f"\n" + "=" * 80)
print("ENSEMBLE CV RESULTS (CLEAN - NO LEAKAGE)")
print("=" * 80)
print(f"  F1:        {ensemble_cv_f1:.4f} ± {ensemble_cv_f1_std:.4f}")
print(f"  Precision: {ensemble_cv_precision:.4f}")
print(f"  Recall:    {ensemble_cv_recall:.4f}")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

test_probas = []
for name in model_names:
    model = cv_results[name]['model']
    proba = model.predict_proba(X_test)[:, 1]
    test_probas.append(proba)
    print(f"  {name:20s}: mean proba = {proba.mean():.4f}")

ensemble_test_proba = np.zeros(len(X_test))
for weight, proba in zip(best_weights, test_probas):
    ensemble_test_proba += weight * proba

ensemble_test_pred = (ensemble_test_proba >= best_threshold).astype(int)

print(f"\n✓ Ensemble test predictions: {ensemble_test_pred.sum()} tickets flagged")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': ensemble_test_pred
})

submission.to_csv('submission_ml_v5_clean.csv', index=False)
print(f"\n✓ Saved: submission_ml_v5_clean.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY - ML PIPELINE v5 CLEAN (NO DATA LEAKAGE)")
print("=" * 80)

print(f"""
CRITICAL FIX:
✓ Removed validation_notes usage (was causing data leakage)
✓ Removed has_any_error_word feature (derived from validation_notes)

Model Performance (10-fold CV):
""")

for name in model_names:
    f1 = cv_results[name]['cv_f1']
    std = cv_results[name]['cv_f1_std']
    print(f"  {name:20s}: F1 = {f1:.4f} ± {std:.4f}")

print(f"\n  {'Ensemble':20s}: F1 = {ensemble_cv_f1:.4f} ± {ensemble_cv_f1_std:.4f}")

print(f"""
Test Predictions:
  Tickets flagged: {ensemble_test_pred.sum()} / {len(test_df)}
  Threshold: {best_threshold:.3f}

This is the CLEAN version with NO data leakage.
Submit: submission_ml_v5_clean.csv
""")

print("=" * 80)
