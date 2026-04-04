"""
CareCaller Hackathon - ML Pipeline v3
Targeted feature engineering based on exact ticket patterns
Target: Val F1 > 0.95
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

print("=" * 70)
print("ML PIPELINE v3 - Targeted Features")
print("=" * 70)

# Load data
train_df = pd.read_csv('../Datasets/csv/hackathon_train.csv')
val_df = pd.read_csv('../Datasets/csv/hackathon_val.csv')
test_df = pd.read_csv('../Datasets/csv/hackathon_test.csv')

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
test_call_ids = test_df['call_id'].copy()

pos_count = train_df['has_ticket'].sum()
neg_count = len(train_df) - pos_count
scale_pos_weight = neg_count / pos_count

# ============================================================================
# HIGHLY TARGETED FEATURE EXTRACTION
# ============================================================================

def extract_ticket_signals(row):
    """Extract the EXACT signals that trigger tickets based on pattern analysis"""
    notes = str(row.get('validation_notes', '')).lower()
    outcome = str(row.get('outcome', '')).lower()
    
    features = {}
    
    # === HIGH SIGNAL PATTERNS (found only in tickets) ===
    
    # 1. Dosage/medical guidance given (130x more likely in tickets)
    features['has_dosage_guidance'] = int('dosage guidance' in notes or 'guidance' in notes)
    features['has_recommendations'] = int('recommendation' in notes)
    
    # 2. Miscategorization detected (70x)
    features['has_miscategorization'] = int('miscategorization' in notes or 'miscategor' in notes)
    
    # 3. Outcome correction by AI (strong signal)
    features['outcome_corrected'] = int('outcome corrected' in notes or 'outcome was corrected' in notes)
    
    # 4. Wrong number scenario mentioned in notes (50x)
    features['notes_wrong_number'] = int('wrong_number' in notes or 'wrong number' in notes)
    
    # 5. STT/transcription errors (30x)
    features['has_stt_error'] = int('stt' in notes or 'speech-to-text' in notes or 'transcription error' in notes)
    
    # 6. Erroneously recorded data
    features['has_erroneous_data'] = int('erroneously' in notes or 'fabricated' in notes or 'incorrect' in notes)
    
    # 7. Patient reschedule/opt-out patterns
    features['wants_reschedule'] = int('reschedule' in notes or 'not right now' in notes or 'maybe later' in notes)
    
    # 8. Error keyword present
    features['has_error_keyword'] = int('error' in notes)
    
    # === OUTCOME-BASED SIGNALS ===
    features['outcome_wrong_number'] = int(outcome == 'wrong_number')
    features['outcome_opted_out'] = int(outcome == 'opted_out')
    features['outcome_escalated'] = int(outcome == 'escalated')
    
    # === WHISPER MISMATCH (strong signal) ===
    features['whisper_mismatch'] = int(row.get('whisper_mismatch_count', 0))
    features['has_whisper_mismatch'] = int(features['whisper_mismatch'] > 0)
    
    # === COMBINED SIGNAL ===
    # Any high-confidence ticket indicator
    features['any_ticket_signal'] = int(
        features['has_dosage_guidance'] or
        features['has_miscategorization'] or
        features['outcome_corrected'] or
        features['has_stt_error'] or
        features['has_erroneous_data'] or
        features['has_whisper_mismatch']
    )
    
    return features


def extract_response_features(responses_json):
    """Extract features from responses"""
    result = {
        'answered_count': 0,
        'empty_answers': 0,
        'answer_ratio': 0.0,
        'suspicious_weight': 0
    }
    
    if pd.isna(responses_json) or responses_json == '':
        return result
    
    try:
        responses = json.loads(responses_json)
        total = len(responses)
        
        for r in responses:
            answer = str(r.get('answer', '')).strip()
            question = str(r.get('question', '')).lower()
            
            if answer == '' or answer.lower() == 'nan':
                result['empty_answers'] += 1
            else:
                result['answered_count'] += 1
                
                # Check for suspicious weight (STT error indicator)
                if 'weight' in question and 'lost' not in question and 'goal' not in question:
                    weight_match = re.search(r'(\d+)', answer)
                    if weight_match:
                        weight = int(weight_match.group(1))
                        if weight < 80 or weight > 500:
                            result['suspicious_weight'] = 1
        
        if total > 0:
            result['answer_ratio'] = result['answered_count'] / total
            
    except:
        pass
    
    return result


def engineer_features_v3(df, label_encoders=None, is_train=False):
    """V3 feature engineering with targeted signals"""
    df = df.copy()
    
    print("  Extracting ticket signals...")
    signal_features = df.apply(extract_ticket_signals, axis=1)
    signal_df = pd.DataFrame(signal_features.tolist(), index=df.index)
    df = pd.concat([df, signal_df], axis=1)
    
    print("  Extracting response features...")
    response_features = df['responses_json'].apply(extract_response_features)
    response_df = pd.DataFrame(response_features.tolist(), index=df.index)
    df = pd.concat([df, response_df], axis=1)
    
    # Basic numeric features
    numeric_cols = ['call_duration', 'attempt_number', 'question_count', 'answered_count',
                    'response_completeness', 'turn_count', 'user_turn_count', 'agent_turn_count',
                    'user_word_count', 'agent_word_count', 'avg_user_turn_words', 
                    'avg_agent_turn_words', 'interruption_count', 'max_time_in_call', 'hour_of_day']
    
    # Label encode categoricals
    categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 
                        'patient_state', 'direction', 'organization_id', 'product_id']
    
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
        'form_submitted'
    ] + categorical_cols
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.fillna(0)
    
    # Ensure numeric
    for col in df.columns:
        if col != 'has_ticket':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                pass
    
    return df, label_encoders


# Process data
print("\nProcessing training data...")
train_processed, label_encs = engineer_features_v3(train_df, is_train=True)

print("Processing validation data...")
val_processed, _ = engineer_features_v3(val_df, label_encoders=label_encs, is_train=False)

print("Processing test data...")
test_processed, _ = engineer_features_v3(test_df, label_encoders=label_encs, is_train=False)

# Prepare X, y
X_train = train_processed.drop('has_ticket', axis=1)
y_train = train_processed['has_ticket'].astype(int)

X_val = val_processed.drop('has_ticket', axis=1)
y_val = val_processed['has_ticket'].astype(int)

X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Remove duplicate columns
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_val = X_val.loc[:, ~X_val.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

# Align columns
common_cols = sorted(list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns)))
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]

print(f"\nFeature count: {len(common_cols)}")
print(f"Features: {list(X_train.columns)}")

# ============================================================================
# TRAIN MODELS
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {
    'LightGBM': LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
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
    val_p = precision_score(y_val, y_val_pred)
    val_r = recall_score(y_val, y_val_pred)
    
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'Val F1': val_f1,
        'Val P': val_p,
        'Val R': val_r
    })
    print(f"  Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, P: {val_p:.4f}, R: {val_r:.4f}")

# ============================================================================
# THRESHOLD TUNING FOR EACH MODEL
# ============================================================================
print("\n" + "=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)

best_overall_f1 = 0
best_model = None
best_threshold = 0.5

for name, model in trained_models.items():
    if hasattr(model, 'predict_proba'):
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"{name}: Best threshold={optimal_threshold:.4f}, F1={optimal_f1:.4f}")
        
        if optimal_f1 > best_overall_f1:
            best_overall_f1 = optimal_f1
            best_model = model
            best_threshold = optimal_threshold
            best_model_name = name

print(f"\nBest: {best_model_name} with F1={best_overall_f1:.4f} at threshold={best_threshold:.4f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))
    
    plt.figure(figsize=(10, 8))
    top_20 = importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'].values, color='steelblue')
    plt.yticks(range(len(top_20)), top_20['feature'].values)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'Top 20 Features ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance_v3.png', dpi=150)
    plt.close()
    print("\nSaved: feature_importance_v3.png")

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_val_proba = best_model.predict_proba(X_val)[:, 1]

y_train_final = (y_train_proba >= best_threshold).astype(int)
y_val_final = (y_val_proba >= best_threshold).astype(int)

print("\n--- TRAINING SET ---")
print(classification_report(y_train, y_train_final, target_names=['No Ticket', 'Has Ticket']))

print("\n--- VALIDATION SET ---")
print(classification_report(y_val, y_val_final, target_names=['No Ticket', 'Has Ticket']))

final_train_f1 = f1_score(y_train, y_train_final)
final_val_f1 = f1_score(y_val, y_val_final)

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': y_test_pred
})
submission.to_csv('submission_ml_v3.csv', index=False)
print(f"Saved: submission_ml_v3.csv")
print(f"Test predictions: {y_test_pred.sum()} tickets flagged")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

print(f"""
+-------------------------+----------+--------+------------+
| Method                  | Train F1 | Val F1 | Test Flags |
+-------------------------+----------+--------+------------+
| Rule-based              |   1.00   |  1.00  |     18     |
| ML v1 (XGBoost)         |   0.97   |  0.90  |     15     |
| ML v2 (LightGBM)        |   1.00   |  0.91  |     17     |
| ML v3 ({best_model_name:15}) |   {final_train_f1:.2f}   |  {final_val_f1:.2f}  |     {y_test_pred.sum()}     |
+-------------------------+----------+--------+------------+

Target: Val F1 > 0.95
Achieved: Val F1 = {final_val_f1:.4f}
""")

if final_val_f1 >= 0.95:
    print("✓ TARGET ACHIEVED!")
elif final_val_f1 >= 0.93:
    print("○ Close! Within 0.02 of target")
else:
    print(f"✗ Gap: {0.95 - final_val_f1:.4f}")

# Show misclassified validation cases
print("\n" + "=" * 70)
print("MISCLASSIFIED VALIDATION CASES")
print("=" * 70)

# False negatives (missed tickets)
fn_mask = (y_val == 1) & (y_val_final == 0)
fn_indices = val_df.index[fn_mask]
print(f"\nFalse Negatives (missed {fn_mask.sum()} tickets):")
for idx in fn_indices:
    row = val_df.loc[idx]
    print(f"  - Outcome: {row['outcome']}, Whisper mismatch: {row['whisper_mismatch_count']}")
    print(f"    Notes: {str(row['validation_notes'])[:100]}...")

# False positives
fp_mask = (y_val == 0) & (y_val_final == 1)
fp_indices = val_df.index[fp_mask]
print(f"\nFalse Positives (incorrectly flagged {fp_mask.sum()}):")
for idx in fp_indices:
    row = val_df.loc[idx]
    print(f"  - Outcome: {row['outcome']}")
    print(f"    Notes: {str(row['validation_notes'])[:100]}...")
