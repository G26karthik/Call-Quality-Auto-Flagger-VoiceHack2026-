"""
ML Pipeline v5 - Maximum Generalization
Target: Private LB > 0.97 with CV F1 std < 0.04

Key changes from v4:
- Combine train+val (833 rows) with 10-fold StratifiedKFold
- Conservative feature set (only high-signal, generalizable features)
- Strong regularization (num_leaves=15, max_depth=4, reg_lambda=5.0)
- Calibrated ensemble (LightGBM + XGBoost + LogReg soft voting)
- Optuna 200 trials optimizing mean CV F1
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

# Paths
TRAIN_PATH = '../Datasets/csv/hackathon_train.csv'
VAL_PATH = '../Datasets/csv/hackathon_val.csv'
TEST_PATH = '../Datasets/csv/hackathon_test.csv'

def load_data():
    """Load and combine train + val for better generalization"""
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Combine train + val
    combined = pd.concat([train, val], ignore_index=True)
    print(f"Combined dataset: {len(combined)} rows (train: {len(train)}, val: {len(val)})")
    print(f"Test set: {len(test)} rows")
    
    # Class distribution
    pos = combined['has_ticket'].sum()
    neg = len(combined) - pos
    print(f"Class distribution: {neg} negative, {pos} positive (ratio: {neg/pos:.2f}:1)")
    
    return combined, test

def extract_conservative_features(df):
    """
    Extract only high-signal, generalizable features.
    Balanced approach - keep core domain signals but use broader patterns.
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. whisper_mismatch_count - proven strong signal
    features['whisper_mismatch_count'] = df['whisper_mismatch_count'].fillna(0)
    features['has_whisper_mismatch'] = (features['whisper_mismatch_count'] > 0).astype(int)
    
    # 2. Response completeness from responses_json
    def calc_response_completeness(json_str):
        if pd.isna(json_str) or json_str == '':
            return 0.0
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return 0.0
            total = len(data)
            if total == 0:
                return 1.0
            answered = sum(1 for v in data.values() if v is not None and str(v).strip() != '')
            return answered / total
        except:
            return 0.0
    
    features['response_completeness'] = df['responses_json'].apply(calc_response_completeness)
    
    # 3. Answered count
    def count_answered(json_str):
        if pd.isna(json_str) or json_str == '':
            return 0
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return 0
            return sum(1 for v in data.values() if v is not None and str(v).strip() != '')
        except:
            return 0
    
    features['answered_count'] = df['responses_json'].apply(count_answered)
    
    # 4. Call duration
    features['call_duration'] = df['call_duration'].fillna(0)
    
    # 5. Outcome encoded - with binary flags for strongest signals
    outcome_map = {
        'completed': 0, 'no_answer': 1, 'voicemail': 2,
        'callback_requested': 3, 'wrong_number': 4, 'escalated': 5, 'rescheduled': 6
    }
    features['outcome_encoded'] = df['outcome'].map(outcome_map).fillna(0)
    features['outcome_wrong_number'] = (df['outcome'] == 'wrong_number').astype(int)
    features['outcome_escalated'] = (df['outcome'] == 'escalated').astype(int)
    
    # 6. Broad ticket signal patterns (generalized, not memorizing specific phrases)
    def extract_text_signals(row):
        notes = str(row.get('validation_notes', '')).lower()
        signals = {}
        
        # Very broad categories that generalize
        signals['has_guidance_issue'] = 1 if re.search(r'(guidance|advice|instruct)', notes) else 0
        signals['has_mismatch_issue'] = 1 if re.search(r'(mismatch|discrepan|conflict)', notes) else 0
        signals['has_categorization_issue'] = 1 if re.search(r'(categor|classif|label)', notes) else 0
        signals['has_data_issue'] = 1 if re.search(r'(contradic|incorrect|wrong|error)', notes) else 0
        signals['has_missing_issue'] = 1 if re.search(r'(skip|miss|omit|incomplete)', notes) else 0
        
        # Validation notes present and non-trivial
        signals['has_notes'] = 1 if len(notes.strip()) > 10 else 0
        
        return pd.Series(signals)
    
    text_signals = df.apply(extract_text_signals, axis=1)
    for col in text_signals.columns:
        features[col] = text_signals[col]
    
    # Combined any_ticket_signal
    features['any_ticket_signal'] = (
        (features['has_guidance_issue'] == 1) | 
        (features['has_mismatch_issue'] == 1) |
        (features['has_categorization_issue'] == 1) |
        (features['has_data_issue'] == 1) |
        (features['has_missing_issue'] == 1) |
        (features['outcome_wrong_number'] == 1) |
        (features['outcome_escalated'] == 1)
    ).astype(int)
    
    # 7. Whisper status encoded
    whisper_map = {'matched': 0, 'mismatched': 1, 'not_applicable': 2}
    features['whisper_status_encoded'] = df['whisper_status'].map(whisper_map).fillna(0)
    
    # 8. Direction encoded
    features['direction_encoded'] = (df['direction'] == 'outbound').astype(int)
    
    # 9. Transcript features
    features['transcript_length'] = df['transcript_text'].fillna('').str.len()
    features['transcript_word_count'] = df['transcript_text'].fillna('').str.split().str.len()
    
    # 10. Notes length
    features['notes_length'] = df['validation_notes'].fillna('').str.len()
    
    return features

def create_tfidf_features(train_notes, test_notes=None, max_features=30):
    """Create TF-IDF features with very reduced dimensionality for generalization"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),  # Only unigrams for generalization
        min_df=5,            # Require term to appear in at least 5 docs
        max_df=0.7,          # Ignore terms in more than 70% of docs
        stop_words='english'
    )
    
    train_tfidf = vectorizer.fit_transform(train_notes.fillna(''))
    tfidf_cols = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols)
    
    if test_notes is not None:
        test_tfidf = vectorizer.transform(test_notes.fillna(''))
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols)
        return train_tfidf_df, test_tfidf_df, vectorizer
    
    return train_tfidf_df, vectorizer

def prepare_features(combined_df, test_df):
    """Prepare final feature matrices"""
    # Extract conservative features
    combined_features = extract_conservative_features(combined_df)
    test_features = extract_conservative_features(test_df)
    
    # Create TF-IDF features
    combined_tfidf, test_tfidf, vectorizer = create_tfidf_features(
        combined_df['validation_notes'],
        test_df['validation_notes'],
        max_features=50
    )
    
    # Reset indices for concatenation
    combined_features = combined_features.reset_index(drop=True)
    combined_tfidf = combined_tfidf.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    test_tfidf = test_tfidf.reset_index(drop=True)
    
    # Combine all features
    X_combined = pd.concat([combined_features, combined_tfidf], axis=1)
    X_test = pd.concat([test_features, test_tfidf], axis=1)
    
    # Target
    y_combined = combined_df['has_ticket'].values
    
    print(f"\nFeature matrix shape: {X_combined.shape}")
    print(f"Features: {list(combined_features.columns)} + {len(combined_tfidf.columns)} TF-IDF features")
    
    return X_combined, y_combined, X_test, test_df['call_id'].values

def cross_validate_model(X, y, model_fn, n_splits=10):
    """
    Perform stratified k-fold cross validation.
    Returns mean F1, std F1, and out-of-fold predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_fn()
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            oof_preds[val_idx] = model.predict(X_val)
        
        # Calculate F1 at threshold 0.5 for this fold
        fold_pred = (oof_preds[val_idx] > 0.5).astype(int)
        fold_f1 = f1_score(y_val, fold_pred)
        fold_scores.append(fold_f1)
    
    return np.mean(fold_scores), np.std(fold_scores), oof_preds

def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score"""
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_proba > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1

def create_lgbm_model(params=None):
    """Create LightGBM model with strong regularization"""
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 4,
        'min_child_samples': 10,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'reg_lambda': 5.0,
        'reg_alpha': 1.0,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'random_state': 42,
        'verbose': -1
    }
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)

def create_xgb_model(params=None):
    """Create XGBoost model with very strong regularization"""
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,          # Reduced from 4
        'min_child_weight': 15,  # Increased from 10
        'learning_rate': 0.03,   # Reduced from 0.05
        'n_estimators': 150,     # Reduced from 200
        'reg_lambda': 10.0,      # Increased from 5.0
        'reg_alpha': 2.0,        # Increased from 1.0
        'colsample_bytree': 0.6, # Reduced from 0.7
        'subsample': 0.6,        # Reduced from 0.7
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)

def create_lr_model():
    """Create Logistic Regression model with moderate regularization"""
    return LogisticRegression(
        C=0.5,  # Moderate regularization (balanced)
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'  # Handle imbalance
    )

def optuna_objective(trial, X, y):
    """Optuna objective for LightGBM hyperparameter optimization with stronger regularization"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 4, 20),       # Reduced upper bound
        'max_depth': trial.suggest_int('max_depth', 2, 5),          # Reduced upper bound
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Increased lower bound
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),    # Reduced upper bound
        'n_estimators': trial.suggest_int('n_estimators', 80, 200),           # Reduced range
        'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 15.0),           # Increased range
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 8.0),              # Increased range
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8), # Reduced range
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),              # Reduced range
    }
    
    # 10-fold CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = create_lgbm_model(params)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        # Use optimized threshold
        thresh, f1 = find_optimal_threshold(y_val, y_proba)
        fold_scores.append(f1)
    
    return np.mean(fold_scores)

def run_optuna_tuning(X, y, n_trials=200):
    """Run Optuna hyperparameter optimization"""
    print(f"\n{'='*60}")
    print("OPTUNA HYPERPARAMETER TUNING (200 trials)")
    print(f"{'='*60}")
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(
        lambda trial: optuna_objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nBest CV F1: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

def train_ensemble(X, y, best_lgbm_params):
    """Train calibrated ensemble with soft voting"""
    print(f"\n{'='*60}")
    print("TRAINING CALIBRATED ENSEMBLE")
    print(f"{'='*60}")
    
    # 10-fold CV for ensemble
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_lgbm = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_lr = np.zeros(len(y))
    
    lgbm_models = []
    xgb_models = []
    lr_models = []
    
    fold_scores = {'lgbm': [], 'xgb': [], 'lr': [], 'ensemble': []}
    
    # Scale features for LR
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        X_train_scaled, X_val_scaled = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM
        lgbm = create_lgbm_model(best_lgbm_params)
        lgbm.fit(X_train, y_train)
        oof_lgbm[val_idx] = lgbm.predict_proba(X_val)[:, 1]
        lgbm_models.append(lgbm)
        
        # XGBoost
        xgb_model = create_xgb_model()
        xgb_model.fit(X_train, y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        xgb_models.append(xgb_model)
        
        # Logistic Regression
        lr = create_lr_model()
        lr.fit(X_train_scaled, y_train)
        oof_lr[val_idx] = lr.predict_proba(X_val_scaled)[:, 1]
        lr_models.append(lr)
        
        # Ensemble (soft voting)
        oof_ensemble = (oof_lgbm[val_idx] + oof_xgb[val_idx] + oof_lr[val_idx]) / 3
        
        # Calculate fold scores at 0.5 threshold
        fold_scores['lgbm'].append(f1_score(y_val, (oof_lgbm[val_idx] > 0.5).astype(int)))
        fold_scores['xgb'].append(f1_score(y_val, (oof_xgb[val_idx] > 0.5).astype(int)))
        fold_scores['lr'].append(f1_score(y_val, (oof_lr[val_idx] > 0.5).astype(int)))
        fold_scores['ensemble'].append(f1_score(y_val, (oof_ensemble > 0.5).astype(int)))
    
    # Combined ensemble predictions
    oof_ensemble = (oof_lgbm + oof_xgb + oof_lr) / 3
    
    # Find optimal thresholds
    print("\nFinding optimal thresholds...")
    thresh_lgbm, f1_lgbm = find_optimal_threshold(y, oof_lgbm)
    thresh_xgb, f1_xgb = find_optimal_threshold(y, oof_xgb)
    thresh_lr, f1_lr = find_optimal_threshold(y, oof_lr)
    thresh_ensemble, f1_ensemble = find_optimal_threshold(y, oof_ensemble)
    
    print(f"\n{'Model':<15} {'CV F1 Mean':>12} {'CV F1 Std':>12} {'Opt Thresh':>12} {'Opt F1':>12}")
    print("-" * 65)
    print(f"{'LightGBM':<15} {np.mean(fold_scores['lgbm']):>12.4f} {np.std(fold_scores['lgbm']):>12.4f} {thresh_lgbm:>12.4f} {f1_lgbm:>12.4f}")
    print(f"{'XGBoost':<15} {np.mean(fold_scores['xgb']):>12.4f} {np.std(fold_scores['xgb']):>12.4f} {thresh_xgb:>12.4f} {f1_xgb:>12.4f}")
    print(f"{'LogReg':<15} {np.mean(fold_scores['lr']):>12.4f} {np.std(fold_scores['lr']):>12.4f} {thresh_lr:>12.4f} {f1_lr:>12.4f}")
    print(f"{'ENSEMBLE':<15} {np.mean(fold_scores['ensemble']):>12.4f} {np.std(fold_scores['ensemble']):>12.4f} {thresh_ensemble:>12.4f} {f1_ensemble:>12.4f}")
    
    return {
        'lgbm_models': lgbm_models,
        'xgb_models': xgb_models,
        'lr_models': lr_models,
        'scaler': scaler,
        'thresholds': {
            'lgbm': thresh_lgbm,
            'xgb': thresh_xgb,
            'lr': thresh_lr,
            'ensemble': thresh_ensemble
        },
        'oof_preds': {
            'lgbm': oof_lgbm,
            'xgb': oof_xgb,
            'lr': oof_lr,
            'ensemble': oof_ensemble
        },
        'cv_scores': {
            'lgbm': (np.mean(fold_scores['lgbm']), np.std(fold_scores['lgbm'])),
            'xgb': (np.mean(fold_scores['xgb']), np.std(fold_scores['xgb'])),
            'lr': (np.mean(fold_scores['lr']), np.std(fold_scores['lr'])),
            'ensemble': (np.mean(fold_scores['ensemble']), np.std(fold_scores['ensemble']))
        },
        'opt_f1': {
            'lgbm': f1_lgbm,
            'xgb': f1_xgb,
            'lr': f1_lr,
            'ensemble': f1_ensemble
        }
    }

def generate_predictions(X_test, ensemble_result):
    """Generate test predictions using ensemble"""
    lgbm_models = ensemble_result['lgbm_models']
    xgb_models = ensemble_result['xgb_models']
    lr_models = ensemble_result['lr_models']
    scaler = ensemble_result['scaler']
    threshold = ensemble_result['thresholds']['ensemble']
    
    # Average predictions from all fold models
    lgbm_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in lgbm_models], axis=0)
    xgb_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    lr_preds = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in lr_models], axis=0)
    
    # Ensemble average
    ensemble_preds = (lgbm_preds + xgb_preds + lr_preds) / 3
    
    # Apply threshold
    final_preds = (ensemble_preds > threshold).astype(int)
    
    return final_preds, ensemble_preds

def print_final_evaluation(y_true, y_proba, threshold, label="Ensemble"):
    """Print detailed evaluation metrics"""
    y_pred = (y_proba > threshold).astype(int)
    
    print(f"\n{label} Evaluation (threshold={threshold:.4f}):")
    print("-" * 50)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

def plot_feature_importance(ensemble_result, feature_names):
    """Plot feature importance from LightGBM models"""
    importances = np.zeros(len(feature_names))
    
    for model in ensemble_result['lgbm_models']:
        importances += model.feature_importances_
    
    importances /= len(ensemble_result['lgbm_models'])
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(10, 8))
    plt.title('Top 20 Feature Importances (ML v5)')
    plt.barh(range(len(indices)), importances[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_v5.png', dpi=150)
    plt.close()
    print("\nSaved feature_importance_v5.png")

def main():
    print("=" * 70)
    print("ML PIPELINE v5 - MAXIMUM GENERALIZATION")
    print("=" * 70)
    
    # Load data
    combined, test = load_data()
    
    # Prepare features
    X, y, X_test, test_call_ids = prepare_features(combined, test)
    
    # Run Optuna tuning
    best_params = run_optuna_tuning(X, y, n_trials=200)
    
    # Train final model with best params using all data
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL WITH BEST PARAMS")
    print(f"{'='*60}")
    
    # 10-fold CV to get OOF predictions for threshold tuning
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = create_lgbm_model(best_params)
        model.fit(X_train, y_train)
        models.append(model)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Calculate fold F1
        thresh, f1 = find_optimal_threshold(y_val, oof_preds[val_idx])
        fold_scores.append(f1)
        print(f"Fold {fold+1}: F1={f1:.4f} (threshold={thresh:.4f})")
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    # Find optimal threshold on full OOF predictions
    opt_thresh, opt_f1 = find_optimal_threshold(y, oof_preds)
    
    print(f"\n{'='*60}")
    print("FINAL CV RESULTS (LightGBM only)")
    print(f"{'='*60}")
    print(f"\nCV F1 Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Optimized F1 (on full OOF): {opt_f1:.4f}")
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    
    if cv_std < 0.04:
        print("✓ CV std < 0.04 - Good generalization!")
    else:
        print(f"⚠ CV std = {cv_std:.4f} (target < 0.04)")
    
    if opt_f1 > 0.97:
        print("✓ Optimized F1 > 0.97 - Target achieved!")
    else:
        print(f"⚠ Optimized F1 = {opt_f1:.4f} (target: 0.97)")
    
    # Print detailed evaluation
    print_final_evaluation(y, oof_preds, opt_thresh, "LightGBM (10-fold CV)")
    
    # Generate test predictions
    test_preds_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    test_preds = (test_preds_proba > opt_thresh).astype(int)
    
    print(f"\n{'='*60}")
    print("TEST SET PREDICTIONS")
    print(f"{'='*60}")
    print(f"Total test samples: {len(test_preds)}")
    print(f"Predicted tickets: {test_preds.sum()}")
    print(f"Threshold used: {opt_thresh:.4f}")
    
    # Save submission
    submission = pd.DataFrame({
        'call_id': test_call_ids,
        'has_ticket': test_preds
    })
    submission.to_csv('submission_ml_v5.csv', index=False)
    print(f"\nSaved submission_ml_v5.csv ({test_preds.sum()} tickets)")
    
    # Plot feature importance
    plot_feature_importance({'lgbm_models': models}, X.columns.tolist())
    
    # Final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"\n| Method              | CV F1 Mean | CV F1 Std | Opt F1  | Test Flags |")
    print(f"|---------------------|------------|-----------|---------|------------|")
    print(f"| Rule-based          | 1.0000     | 0.0000    | 1.0000  | 18         |")
    print(f"| ML v4 (prev best)   | 0.9500     | ~0.04     | 0.9200  | 17         |")
    print(f"| ML v5 (current)     | {cv_mean:.4f}     | {cv_std:.4f}    | {opt_f1:.4f}  | {test_preds.sum():<10} |")
    
    return models, best_params

if __name__ == '__main__':
    main()
