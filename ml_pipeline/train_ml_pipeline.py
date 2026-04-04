"""
CareCaller Hackathon - ML Pipeline for Binary Classification of has_ticket
Full pipeline: Feature Engineering, Model Training, Hyperparameter Tuning, Threshold Optimization
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    classification_report, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

train_df = pd.read_csv('../Datasets/csv/hackathon_train.csv')
val_df = pd.read_csv('../Datasets/csv/hackathon_val.csv')
test_df = pd.read_csv('../Datasets/csv/hackathon_test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# Store call_ids for submission
test_call_ids = test_df['call_id'].copy()

# Check class distribution
print(f"\nTrain has_ticket distribution:")
print(train_df['has_ticket'].value_counts())
pos_count = train_df['has_ticket'].sum()
neg_count = len(train_df) - pos_count
scale_pos_weight = neg_count / pos_count
print(f"\nClass imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
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

# Categorical columns to label encode
categorical_cols = ['outcome', 'whisper_status', 'day_of_week', 'cycle_status', 'patient_state', 'direction', 'organization_id', 'product_id']

# Feature engineering functions
def extract_transcript_features(text):
    """Extract features from transcript_text"""
    if pd.isna(text) or text == '':
        return 0, 0, 0.0
    
    text_length = len(str(text))
    word_count = len(str(text).split())
    
    # Count agent vs user turns
    agent_pattern = r'\[AGENT\]:'
    user_pattern = r'\[USER\]:'
    agent_count = len(re.findall(agent_pattern, str(text)))
    user_count = len(re.findall(user_pattern, str(text)))
    
    if agent_count > 0:
        agent_user_ratio = user_count / agent_count
    else:
        agent_user_ratio = 0.0
    
    return text_length, word_count, agent_user_ratio

def extract_response_features(responses_json):
    """Extract features from responses_json"""
    if pd.isna(responses_json) or responses_json == '':
        return 0, 0, 0
    
    try:
        responses = json.loads(responses_json)
        non_null_count = sum(1 for r in responses if r.get('answer', '').strip() != '')
        
        yes_count = 0
        no_count = 0
        for r in responses:
            answer = str(r.get('answer', '')).lower().strip()
            if answer in ['yes', 'yeah', 'yep', 'y']:
                yes_count += 1
            elif answer in ['no', 'nope', 'n']:
                no_count += 1
        
        return non_null_count, yes_count, no_count
    except:
        return 0, 0, 0

def engineer_features(df, tfidf_vectorizer=None, label_encoders=None, is_train=False):
    """Complete feature engineering pipeline"""
    df = df.copy()
    
    # Extract transcript features
    print("  Extracting transcript features...")
    transcript_features = df['transcript_text'].apply(extract_transcript_features)
    df['transcript_length'] = [f[0] for f in transcript_features]
    df['transcript_word_count'] = [f[1] for f in transcript_features]
    df['agent_user_ratio'] = [f[2] for f in transcript_features]
    
    # Extract response features
    print("  Extracting response features...")
    response_features = df['responses_json'].apply(extract_response_features)
    df['non_null_answers'] = [f[0] for f in response_features]
    df['yes_responses'] = [f[1] for f in response_features]
    df['no_responses'] = [f[2] for f in response_features]
    
    # TF-IDF on validation_notes
    print("  Creating TF-IDF features from validation_notes...")
    df['validation_notes'] = df['validation_notes'].fillna('')
    
    if is_train:
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['validation_notes'])
    else:
        tfidf_matrix = tfidf_vectorizer.transform(df['validation_notes'])
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
        index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Label encode categorical columns
    print("  Label encoding categorical columns...")
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Drop columns
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors='ignore')
    
    # Drop text columns after feature extraction
    df = df.drop(columns=['transcript_text', 'validation_notes', 'responses_json', 'whisper_transcript'], errors='ignore')
    
    # Fill remaining NaN with 0
    df = df.fillna(0)
    
    return df, tfidf_vectorizer, label_encoders

# Apply feature engineering
print("\nProcessing training data...")
train_processed, tfidf_vec, label_encs = engineer_features(train_df, is_train=True)

print("\nProcessing validation data...")
val_processed, _, _ = engineer_features(val_df, tfidf_vectorizer=tfidf_vec, label_encoders=label_encs, is_train=False)

print("\nProcessing test data...")
test_processed, _, _ = engineer_features(test_df, tfidf_vectorizer=tfidf_vec, label_encoders=label_encs, is_train=False)

# Separate features and target
X_train = train_processed.drop('has_ticket', axis=1)
y_train = train_processed['has_ticket'].astype(int)

X_val = val_processed.drop('has_ticket', axis=1)
y_val = val_processed['has_ticket'].astype(int)

# Test has no target
X_test = test_processed.drop('has_ticket', axis=1, errors='ignore')

# Ensure same columns in all sets
common_cols = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]

print(f"\nFeature count: {len(common_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================================
# STEP 3: HANDLE CLASS IMBALANCE - SMOTE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: HANDLING CLASS IMBALANCE WITH SMOTE")
print("=" * 70)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_smote.shape}")
print(f"SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

# ============================================================================
# STEP 4: TRAIN MULTIPLE MODELS AND COMPARE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAINING MULTIPLE MODELS")
print("=" * 70)

models = {
    'XGBoost (scale_pos_weight)': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    ),
    'XGBoost (SMOTE)': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
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

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use SMOTE data for SMOTE model, original for others
    if 'SMOTE' in name:
        model.fit(X_train_smote, y_train_smote)
    else:
        model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
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
    
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

# Show results table
results_df = pd.DataFrame(results)
print("\n" + "-" * 70)
print("MODEL COMPARISON RESULTS:")
print("-" * 70)
print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.loc[results_df['Val F1'].idxmax(), 'Model']
print(f"\nBest model: {best_model_name}")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: HYPERPARAMETER TUNING WITH OPTUNA")
print("=" * 70)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

# Run Optuna optimization
print("Running Optuna hyperparameter search (50 trials)...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest F1 Score: {study.best_value:.4f}")
print(f"Best Parameters: {study.best_params}")

# Train final model with best params
best_params = study.best_params
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['eval_metric'] = 'logloss'

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# ============================================================================
# STEP 6: THRESHOLD TUNING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: THRESHOLD TUNING")
print("=" * 70)

# Get probability predictions
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Calculate F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
optimal_f1 = f1_scores[optimal_idx]

print(f"Default threshold (0.5) F1: {f1_score(y_val, (y_val_proba >= 0.5).astype(int)):.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Optimal F1 score: {optimal_f1:.4f}")

# Plot precision-recall curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recalls, precisions, 'b-', linewidth=2)
plt.scatter([recalls[optimal_idx]], [precisions[optimal_idx]], color='red', s=100, zorder=5, label=f'Optimal (thresh={optimal_threshold:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_curve.png")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 10))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'].values, color='steelblue')
plt.yticks(range(len(top_20)), top_20['feature'].values)
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ============================================================================
# STEP 8: FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: FINAL EVALUATION")
print("=" * 70)

# Final predictions with optimal threshold
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_train_final = (y_train_proba >= optimal_threshold).astype(int)
y_val_final = (y_val_proba >= optimal_threshold).astype(int)

print("\n--- TRAINING SET CLASSIFICATION REPORT ---")
print(classification_report(y_train, y_train_final, target_names=['No Ticket (0)', 'Has Ticket (1)']))

print("\n--- VALIDATION SET CLASSIFICATION REPORT ---")
print(classification_report(y_val, y_val_final, target_names=['No Ticket (0)', 'Has Ticket (1)']))

# Final metrics
train_f1_final = f1_score(y_train, y_train_final)
val_f1_final = f1_score(y_val, y_val_final)

print(f"\nFinal Train F1: {train_f1_final:.4f}")
print(f"Final Val F1: {val_f1_final:.4f}")

# ============================================================================
# STEP 9: GENERATE SUBMISSION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATING SUBMISSION")
print("=" * 70)

# Predict on test set
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'call_id': test_call_ids,
    'has_ticket': y_test_pred
})

submission.to_csv('submission_ml.csv', index=False)
print(f"Saved: submission_ml.csv")
print(f"Test predictions: {y_test_pred.sum()} tickets flagged out of {len(y_test_pred)}")

# ============================================================================
# FINAL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

# Load rule-based submission for comparison
try:
    rule_based = pd.read_csv('../submission.csv')
    rule_based_flags = rule_based['has_ticket'].sum()
except:
    rule_based_flags = 18  # From user's note

comparison_table = f"""
+------------------+----------+--------+------------+
| Method           | Train F1 | Val F1 | Test Flags |
+------------------+----------+--------+------------+
| Rule-based       |   1.00   |  1.00  |     {rule_based_flags}     |
| Best ML Model    |   {train_f1_final:.2f}   |  {val_f1_final:.2f}  |     {y_test_pred.sum()}     |
+------------------+----------+--------+------------+
"""
print(comparison_table)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print(f"""
Summary:
- Best model: XGBoost with Optuna-tuned hyperparameters
- Optimal threshold: {optimal_threshold:.4f}
- Train F1: {train_f1_final:.4f}
- Val F1: {val_f1_final:.4f}
- Test tickets flagged: {y_test_pred.sum()}

Files generated:
- submission_ml.csv (test predictions)
- feature_importance.png (top 20 features)
- precision_recall_curve.png (threshold analysis)
""")
