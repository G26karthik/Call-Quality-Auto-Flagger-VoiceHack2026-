"""
CareCaller Hackathon - Enhanced ML Pipeline v2
Target: Val F1 > 0.95
Improvements: Domain-specific features, LightGBM, Stacking
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    classification_report, precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=" * 70)
print("ENHANCED ML PIPELINE v2 - Target: Val F1 > 0.95")
print("=" * 70)

train_df = pd.read_csv('../Datasets/csv/hackathon_train.csv')
val_df = pd.read_csv('../Datasets/csv/hackathon_val.csv')
test_df = pd.read_csv('../Datasets/csv/hackathon_test.csv')

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

test_call_ids = test_df['call_id'].copy()

pos_count = train_df['has_ticket'].sum()
neg_count = len(train_df) - pos_count
scale_pos_weight = neg_count / pos_count
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

# ============================================================================
# STEP 2: ENHANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: ENHANCED FEATURE ENGINEERING")
print("=" * 70)

# Columns to drop (label leakage and identifiers)
drop_cols = [
    'call_id', 'patient_name_anon', 'attempted_at', 'scheduled_at',
    'ticket_has_reason', 'ticket_priority', 'ticket_status',
    'ticket_initial_notes', 'ticket_resolution_notes',
    'ticket_cat_audio_issue', 'ticket_cat_audio_notes',
    'ticket_cat_elevenlabs', 'ticket_cat_elevenlabs_notes',
    'ticket_cat_openai', 'ticket_cat_openai_notes',
    'ticket_cat_supabase', 'ticket_cat_supabase_notes',
    'ticket_cat_scheduler_aws', 'ticket_cat_scheduler_aws_notes',
    'ticket_cat_other', 'ticket_cat_other_notes',
    'ticket_raised_at', 'ticket_resolved_at'
]

categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                    'patient_state', 'direction', 'organization_id', 'product_id']

# ============================================================================
# NEW: Domain-specific feature extraction functions
# ============================================================================

def extract_validation_notes_features(notes):
    """Extract binary flags for 6 error categories + error indicator"""
    if pd.isna(notes) or notes == '':
        return {
            'has_stt_error': 0,
            'has_medical_advice': 0,
            'has_outcome_mismatch': 0,
            'has_data_contradiction': 0,
            'has_skipped_questions': 0,
            'has_wrong_number': 0,
            'has_any_error': 0,
            'has_whisper_mismatch': 0,
            'has_escalation': 0,
            'has_incomplete': 0,
            'error_keyword_count': 0
        }
    
    notes_lower = notes.lower()
    
    # STT/Audio errors
    stt_keywords = ['stt', 'speech-to-text', 'transcription error', 'audio issue', 
                    'misheard', 'garbled', 'unclear audio', 'audio quality']
    has_stt = int(any(kw in notes_lower for kw in stt_keywords))
    
    # Medical advice issues
    medical_keywords = ['medical advice', 'dosage', 'medication recommendation', 
                        'treatment suggestion', 'health advice', 'medical guidance',
                        'prescri', 'diagnosis']
    has_medical = int(any(kw in notes_lower for kw in medical_keywords))
    
    # Outcome mismatch
    outcome_keywords = ['outcome mismatch', 'incorrect outcome', 'wrong outcome',
                        'outcome should be', 'marked as', 'but actually']
    has_outcome_mismatch = int(any(kw in notes_lower for kw in outcome_keywords))
    
    # Data contradiction
    contradiction_keywords = ['contradiction', 'inconsistent', 'mismatch', 'discrepancy',
                              'doesn\'t match', 'does not match', 'conflicts with']
    has_contradiction = int(any(kw in notes_lower for kw in contradiction_keywords))
    
    # Skipped questions
    skipped_keywords = ['skipped', 'missing question', 'question not asked', 
                        'incomplete questionnaire', 'questions were asked before']
    has_skipped = int(any(kw in notes_lower for kw in skipped_keywords))
    
    # Wrong number/person
    wrong_keywords = ['wrong number', 'wrong person', 'not the patient', 
                      'different person', 'incorrect patient', 'wrong patient']
    has_wrong = int(any(kw in notes_lower for kw in wrong_keywords))
    
    # Whisper mismatch
    whisper_keywords = ['whisper', 'mismatch', 'verification']
    has_whisper = int(any(kw in notes_lower for kw in whisper_keywords))
    
    # Escalation indicators
    escalation_keywords = ['escalat', 'human intervention', 'support person', 'medical concern']
    has_escalation = int(any(kw in notes_lower for kw in escalation_keywords))
    
    # Incomplete call
    incomplete_keywords = ['incomplete', 'disconnection', 'hung up', 'dropped', 'cut off']
    has_incomplete = int(any(kw in notes_lower for kw in incomplete_keywords))
    
    # General error keywords
    error_keywords = ['error', 'issue', 'problem', 'fail', 'incorrect', 'wrong', 
                      'invalid', 'missing', 'suspicious', 'unusual', 'concern']
    error_count = sum(1 for kw in error_keywords if kw in notes_lower)
    
    has_any_error = int(has_stt or has_medical or has_outcome_mismatch or 
                        has_contradiction or has_skipped or has_wrong or 
                        has_whisper or has_escalation or error_count > 0)
    
    return {
        'has_stt_error': has_stt,
        'has_medical_advice': has_medical,
        'has_outcome_mismatch': has_outcome_mismatch,
        'has_data_contradiction': has_contradiction,
        'has_skipped_questions': has_skipped,
        'has_wrong_number': has_wrong,
        'has_any_error': has_any_error,
        'has_whisper_mismatch': has_whisper,
        'has_escalation': has_escalation,
        'has_incomplete': has_incomplete,
        'error_keyword_count': error_count
    }


def extract_response_features(responses_json):
    """Enhanced features from responses_json"""
    result = {
        'total_questions': 0,
        'answered_count': 0,
        'empty_answers': 0,
        'yes_count': 0,
        'no_count': 0,
        'answer_ratio': 0.0,
        'suspicious_weight': 0,
        'has_numeric_weight': 0,
        'weight_value': 0
    }
    
    if pd.isna(responses_json) or responses_json == '':
        return result
    
    try:
        responses = json.loads(responses_json)
        result['total_questions'] = len(responses)
        
        for r in responses:
            answer = str(r.get('answer', '')).strip()
            question = str(r.get('question', '')).lower()
            
            if answer == '' or answer.lower() == 'nan':
                result['empty_answers'] += 1
            else:
                result['answered_count'] += 1
                
                answer_lower = answer.lower()
                if answer_lower in ['yes', 'yeah', 'yep', 'y', 'sure', 'correct']:
                    result['yes_count'] += 1
                elif answer_lower in ['no', 'nope', 'n', 'not really', 'none']:
                    result['no_count'] += 1
                
                # Check for weight question
                if 'weight' in question and 'lost' not in question and 'goal' not in question:
                    # Try to extract numeric weight
                    weight_match = re.search(r'(\d+)', answer)
                    if weight_match:
                        weight = int(weight_match.group(1))
                        result['has_numeric_weight'] = 1
                        result['weight_value'] = weight
                        # Suspicious if under 100 lbs (likely STT error)
                        if weight < 100:
                            result['suspicious_weight'] = 1
        
        if result['total_questions'] > 0:
            result['answer_ratio'] = result['answered_count'] / result['total_questions']
            
    except:
        pass
    
    return result


def extract_transcript_features(text, outcome=None):
    """Enhanced features from transcript_text"""
    result = {
        'transcript_length': 0,
        'transcript_word_count': 0,
        'agent_user_ratio': 0.0,
        'patient_confirmed_identity': 0,
        'agent_mentions_dosage': 0,
        'agent_mentions_medical': 0,
        'has_goodbye': 0,
        'has_callback_request': 0,
        'user_engaged': 0,
        'outcome_transcript_mismatch': 0
    }
    
    if pd.isna(text) or text == '':
        return result
    
    text_str = str(text)
    text_lower = text_str.lower()
    
    result['transcript_length'] = len(text_str)
    result['transcript_word_count'] = len(text_str.split())
    
    # Count agent vs user turns
    agent_count = len(re.findall(r'\[AGENT\]:', text_str))
    user_count = len(re.findall(r'\[USER\]:', text_str))
    
    if agent_count > 0:
        result['agent_user_ratio'] = user_count / agent_count
    
    # Patient confirmed identity
    identity_patterns = [
        r'\[USER\]:\s*(yes|yeah|that\'s me|this is|speaking|correct)',
        r'\[USER\]:\s*yes,?\s*(this is|that\'s)',
    ]
    result['patient_confirmed_identity'] = int(any(re.search(p, text_lower) for p in identity_patterns))
    
    # Agent mentions dosage/medication
    dosage_patterns = ['mg', 'dosage', 'dose', 'injection', 'medication', 'semaglutide', 
                       'liraglutide', 'phentermine', 'refill']
    result['agent_mentions_dosage'] = int(any(p in text_lower for p in dosage_patterns))
    
    # Agent gives medical advice (concerning)
    medical_advice_patterns = ['you should take', 'i recommend', 'increase your', 
                               'decrease your', 'try taking', 'medical advice']
    result['agent_mentions_medical'] = int(any(p in text_lower for p in medical_advice_patterns))
    
    # Call ended properly
    goodbye_patterns = ['take care', 'goodbye', 'have a great', 'thank you for']
    result['has_goodbye'] = int(any(p in text_lower for p in goodbye_patterns))
    
    # Callback request
    callback_patterns = ['call us back', 'call back', 'try again later']
    result['has_callback_request'] = int(any(p in text_lower for p in callback_patterns))
    
    # User engagement (did user speak at all?)
    result['user_engaged'] = int(user_count > 0)
    
    # Outcome-transcript mismatch detection
    if outcome:
        outcome_lower = str(outcome).lower()
        
        # If outcome is 'completed' but user never spoke
        if outcome_lower == 'completed' and user_count == 0:
            result['outcome_transcript_mismatch'] = 1
        
        # If outcome is 'voicemail' but user responded
        if outcome_lower == 'voicemail' and user_count > 2:
            result['outcome_transcript_mismatch'] = 1
            
        # If outcome is 'completed' but has callback request
        if outcome_lower == 'completed' and result['has_callback_request']:
            result['outcome_transcript_mismatch'] = 1
    
    return result


def engineer_features_v2(df, label_encoders=None, is_train=False):
    """Enhanced feature engineering pipeline v2"""
    df = df.copy()
    
    # 1. Extract validation_notes features (domain-specific)
    print("  Extracting validation_notes features (6 error categories)...")
    notes_features = df['validation_notes'].apply(extract_validation_notes_features)
    notes_df = pd.DataFrame(notes_features.tolist(), index=df.index)
    df = pd.concat([df, notes_df], axis=1)
    
    # 2. Extract response features (enhanced)
    print("  Extracting response features (enhanced)...")
    response_features = df['responses_json'].apply(extract_response_features)
    response_df = pd.DataFrame(response_features.tolist(), index=df.index)
    df = pd.concat([df, response_df], axis=1)
    
    # 3. Extract transcript features (enhanced)
    print("  Extracting transcript features (enhanced)...")
    transcript_features = df.apply(
        lambda row: extract_transcript_features(row['transcript_text'], row['outcome']), 
        axis=1
    )
    transcript_df = pd.DataFrame(transcript_features.tolist(), index=df.index)
    df = pd.concat([df, transcript_df], axis=1)
    
    # 4. Label encode categorical columns
    print("  Label encoding categorical columns...")
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('unknown')
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # 5. Drop columns
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors='ignore')
    df = df.drop(columns=['transcript_text', 'validation_notes', 'responses_json', 'whisper_transcript'], errors='ignore')
    
    # 6. Fill NaN
    df = df.fillna(0)
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    return df, label_encoders


# Apply enhanced feature engineering
print("\nProcessing training data...")
train_processed, label_encs = engineer_features_v2(train_df, is_train=True)

print("\nProcessing validation data...")
val_processed, _ = engineer_features_v2(val_df, label_encoders=label_encs, is_train=False)

print("\nProcessing test data...")
test_processed, _ = engineer_features_v2(test_df, label_encoders=label_encs, is_train=False)

# Separate features and target
X_train = train_processed.drop('has_ticket', axis=1)
y_train = train_processed['has_ticket'].astype(int)

X_val = val_processed.drop('has_ticket', axis=1)
y_val = val_processed['has_ticket'].astype(int)

X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Ensure same columns and remove duplicates
common_cols = sorted(list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns)))
X_train = X_train.loc[:, ~X_train.columns.duplicated()][common_cols]
X_val = X_val.loc[:, ~X_val.columns.duplicated()][common_cols]
X_test = X_test.loc[:, ~X_test.columns.duplicated()][common_cols]

# Ensure all numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"\nFeature count: {len(common_cols)}")
print(f"Key new features: has_stt_error, has_medical_advice, suspicious_weight, patient_confirmed_identity, etc.")

# ============================================================================
# STEP 3: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: TRAINING MULTIPLE MODELS")
print("=" * 70)

models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'Val F1': val_f1,
        'Val Precision': val_precision,
        'Val Recall': val_recall
    })
    
    print(f"  Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}")

# ============================================================================
# STEP 4: STACKING ENSEMBLE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: STACKING ENSEMBLE")
print("=" * 70)

# Create stacking classifier
estimators = [
    ('xgb', XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight, 
                          random_state=42, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight, 
                            random_state=42, verbose=-1)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    passthrough=True  # Include original features
)

print("Training Stacking Ensemble (XGB + LGBM + RF → LR)...")
stacking_model.fit(X_train, y_train)
trained_models['Stacking'] = stacking_model

y_train_pred = stacking_model.predict(X_train)
y_val_pred = stacking_model.predict(X_val)

train_f1 = f1_score(y_train, y_train_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)

results.append({
    'Model': 'Stacking Ensemble',
    'Train F1': train_f1,
    'Val F1': val_f1,
    'Val Precision': val_precision,
    'Val Recall': val_recall
})

print(f"  Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}")

# ============================================================================
# STEP 5: OPTUNA TUNING FOR BEST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: OPTUNA HYPERPARAMETER TUNING (LightGBM)")
print("=" * 70)

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

print("Running Optuna search (100 trials)...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_lgbm, n_trials=100, show_progress_bar=True)

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train tuned LightGBM
best_params = study.best_params
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['verbose'] = -1

tuned_lgbm = LGBMClassifier(**best_params)
tuned_lgbm.fit(X_train, y_train)
trained_models['LightGBM (Tuned)'] = tuned_lgbm

y_train_pred = tuned_lgbm.predict(X_train)
y_val_pred = tuned_lgbm.predict(X_val)

train_f1 = f1_score(y_train, y_train_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)

results.append({
    'Model': 'LightGBM (Tuned)',
    'Train F1': train_f1,
    'Val F1': val_f1,
    'Val Precision': val_precision,
    'Val Recall': val_recall
})

print(f"\nTuned LightGBM - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

# ============================================================================
# STEP 6: THRESHOLD TUNING FOR BEST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: THRESHOLD TUNING")
print("=" * 70)

# Find best model
results_df = pd.DataFrame(results)
best_idx = results_df['Val F1'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"Best model: {best_model_name}")

# Get the actual best model
if 'Tuned' in best_model_name:
    best_model = tuned_lgbm
elif 'Stacking' in best_model_name:
    best_model = stacking_model
else:
    best_model = trained_models[best_model_name]

# Threshold tuning
y_val_proba = best_model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
optimal_f1 = f1_scores[optimal_idx]

print(f"Default (0.5) F1: {f1_score(y_val, (y_val_proba >= 0.5).astype(int)):.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Optimal F1: {optimal_f1:.4f}")

# Add threshold-tuned result
y_val_tuned = (y_val_proba >= optimal_threshold).astype(int)
results.append({
    'Model': f'{best_model_name} + Threshold',
    'Train F1': f1_score(y_train, (best_model.predict_proba(X_train)[:, 1] >= optimal_threshold).astype(int)),
    'Val F1': optimal_f1,
    'Val Precision': precision_score(y_val, y_val_tuned),
    'Val Recall': recall_score(y_val, y_val_tuned)
})

# ============================================================================
# STEP 7: RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Val F1', ascending=False)
print(results_df.to_string(index=False))

# Best overall
best_result = results_df.iloc[0]
print(f"\n{'='*70}")
print(f"BEST MODEL: {best_result['Model']}")
print(f"Val F1: {best_result['Val F1']:.4f}")
print(f"Val Precision: {best_result['Val Precision']:.4f}")
print(f"Val Recall: {best_result['Val Recall']:.4f}")
print(f"{'='*70}")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (LightGBM Tuned)")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': tuned_lgbm.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 Most Important Features:")
print(feature_importance.head(25).to_string(index=False))

# Plot
plt.figure(figsize=(12, 10))
top_25 = feature_importance.head(25)
plt.barh(range(len(top_25)), top_25['importance'].values, color='steelblue')
plt.yticks(range(len(top_25)), top_25['feature'].values)
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top 25 Feature Importances (LightGBM Tuned)')
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: feature_importance_v2.png")

# ============================================================================
# STEP 9: FINAL CLASSIFICATION REPORTS
# ============================================================================
print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)

# Use best model with optimal threshold
y_train_final = (best_model.predict_proba(X_train)[:, 1] >= optimal_threshold).astype(int)
y_val_final = (best_model.predict_proba(X_val)[:, 1] >= optimal_threshold).astype(int)

print("\n--- TRAINING SET ---")
print(classification_report(y_train, y_train_final, target_names=['No Ticket', 'Has Ticket']))

print("\n--- VALIDATION SET ---")
print(classification_report(y_val, y_val_final, target_names=['No Ticket', 'Has Ticket']))

# ============================================================================
# STEP 10: GENERATE SUBMISSION
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': y_test_pred
})

submission.to_csv('submission_ml_v2.csv', index=False)
print(f"Saved: submission_ml_v2.csv")
print(f"Test predictions: {y_test_pred.sum()} tickets flagged out of {len(y_test_pred)}")

# ============================================================================
# FINAL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

final_train_f1 = f1_score(y_train, y_train_final)
final_val_f1 = f1_score(y_val, y_val_final)

print(f"""
+-------------------------+----------+--------+------------+
| Method                  | Train F1 | Val F1 | Test Flags |
+-------------------------+----------+--------+------------+
| Rule-based              |   1.00   |  1.00  |     18     |
| ML v1 (XGBoost)         |   0.97   |  0.90  |     15     |
| ML v2 ({best_result['Model'][:15]:<15}) |   {final_train_f1:.2f}   |  {final_val_f1:.2f}  |     {y_test_pred.sum()}     |
+-------------------------+----------+--------+------------+

Target: Val F1 > 0.95
Achieved: Val F1 = {final_val_f1:.4f}
""")

if final_val_f1 > 0.95:
    print("✓ TARGET ACHIEVED!")
else:
    print(f"✗ Target not reached. Gap: {0.95 - final_val_f1:.4f}")
