import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from warnings import filterwarnings

filterwarnings("ignore")

def load_data():
    train = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_train.csv")
    val = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_val.csv")
    test = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_test.csv")
    # Using the /mnt/user-data/uploads/ or local path
    return train, val, test

def engineer_features(df):
    df_feat = df.copy()
    
    # Text features
    df_feat['validation_notes'] = df_feat['validation_notes'].fillna('')
    df_feat['has_outcome_correction'] = df_feat['validation_notes'].str.contains('corrected call_outcome|outcome corrected', case=False, regex=True).astype(int)
    df_feat['has_labeled_but'] = df_feat['validation_notes'].str.contains('labeled outcome as.*but', case=False, regex=True).astype(int)
    df_feat['notes_length'] = df_feat['validation_notes'].str.len()
    
    # Engineered features
    df_feat['duration_per_answer'] = df_feat['call_duration'] / (df_feat['answered_count'] + 1)
    df_feat['is_completed'] = (df_feat['outcome'] == 'completed').astype(int)
    df_feat['is_incomplete'] = (df_feat['outcome'] == 'incomplete').astype(int)
    df_feat['is_wrong_number'] = (df_feat['outcome'] == 'wrong_number').astype(int)
    df_feat['answer_gap'] = df_feat['question_count'] - df_feat['answered_count']
    df_feat['high_duration'] = (df_feat['call_duration'] > 150).astype(int)
    df_feat['form_submitted'] = df_feat['form_submitted'].astype(int) if df_feat['form_submitted'].dtype == bool else df_feat['form_submitted'].fillna(0).astype(int)

    return df_feat

def main():
    train, val, test = load_data()
    
    # Combine train + val for CV
    train_val_combined = pd.concat([train, val]).reset_index(drop=True)
    
    # Encode label 'outcome'
    le = LabelEncoder()
    le.fit(train_val_combined['outcome'].astype(str))
    
    for df in [train_val_combined, val, test]:
        df['outcome_encoded'] = le.transform(df['outcome'].astype(str))
        
    train_val_combined = engineer_features(train_val_combined)
    val_set = engineer_features(val)
    test_set = engineer_features(test)
    
    features = [
        'call_duration', 'answered_count', 'response_completeness',
        'pipeline_mismatch_count', 'outcome_encoded', 'attempt_number',
        'form_submitted', 'interruption_count', 'turn_count',
        'user_word_count', 'agent_word_count', 'avg_user_turn_words',
        'avg_agent_turn_words', 'duration_per_answer', 'is_completed',
        'is_incomplete', 'is_wrong_number', 'answer_gap', 'high_duration',
        'has_outcome_correction', 'has_labeled_but', 'notes_length'
    ]
    
    X = train_val_combined[features]
    y = train_val_combined['has_ticket'].fillna(0).astype(int)
    
    X_val = val_set[features]
    y_val = val_set['has_ticket'].fillna(0).astype(int)
    
    X_test = test_set[features]
    test_ids = test_set['call_id']
    
    # Handing NaNs for Logistic/RF
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Models
    models = {
        'LightGBM': LGBMClassifier(scale_pos_weight=68, random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=68, random_state=42, eval_metric="logloss"),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'Logistic Regression': ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
    }
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    print("| Model | CV F1 | CV Recall | CV Precision | Val F1 | Val Recall | Val Precision |")
    print("|---|---|---|---|---|---|---|")
    
    best_model_name = "LightGBM" # Default
    best_f1 = -1
    models_fitted = {}
    models_oof_probs = {}
    
    for name, model in models.items():
        oof_preds = np.zeros(len(X))
        oof_probs = np.zeros(len(X))
        
        cv_f1s = []
        cv_recs = []
        cv_precs = []
        
        X_used = X_imputed if name in ['Random Forest', 'Logistic Regression'] else X
        X_val_used = X_val_imputed if name in ['Random Forest', 'Logistic Regression'] else X_val
        X_test_used = X_test_imputed if name in ['Random Forest', 'Logistic Regression'] else X_test
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_used, y)):
            X_tr, y_tr = X_used.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_used.iloc[val_idx], y.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs
            preds = (probs > 0.5).astype(int)
            
            cv_f1s.append(f1_score(y_va, preds))
            cv_recs.append(recall_score(y_va, preds))
            cv_precs.append(precision_score(y_va, preds, zero_division=0))
            
        # Refit on just train to get pure Val scores
        X_train_only = X_used.iloc[:len(train)]
        y_train_only = y.iloc[:len(train)]
        model.fit(X_train_only, y_train_only)
        
        val_probs = model.predict_proba(X_val_used)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        
        val_f1 = f1_score(y_val, val_preds)
        val_rec = recall_score(y_val, val_preds)
        val_prec = precision_score(y_val, val_preds, zero_division=0)
        
        print(f"| {name} | {np.mean(cv_f1s):.4f} | {np.mean(cv_recs):.4f} | {np.mean(cv_precs):.4f} | {val_f1:.4f} | {val_rec:.4f} | {val_prec:.4f} |")
        
        # Finally refit on train_val for final predictions
        model.fit(X_used, y)
        models_fitted[name] = model
        models_oof_probs[name] = oof_probs

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_name = name
            
    print(f"\nBest Model by Val F1 (>0.5 threshold): {best_model_name}")
    
# Threshold Tuning using out-of-fold predictions
    best_model = models_fitted[best_model_name]
    best_oof_probs = models_oof_probs[best_model_name]

    precisions, recalls, thresholds = precision_recall_curve(y, best_oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Max F1 threshold
    max_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = thresholds[max_f1_idx] if max_f1_idx < len(thresholds) else 0.5
    
    # Recall >= 0.70 threshold with best precision
    valid_recall_mask = recalls[:-1] >= 0.70 # Exclude last as recall=0
    if np.any(valid_recall_mask):
        best_prec_for_recall_idx = np.argmax(precisions[:-1][valid_recall_mask])
        true_idx = np.where(valid_recall_mask)[0][best_prec_for_recall_idx]
        best_recall_thresh = thresholds[true_idx]
    else:
        best_recall_thresh = best_f1_thresh
        
    print(f"\nThreshold tuning for {best_model_name}:")
    print(f"Optimal Threshold for F1: {best_f1_thresh:.4f}")
    print(f"Optimal Threshold for Recall >= 0.70: {best_recall_thresh:.4f}")
    
    # Generate Submissions
    X_test_used = X_test_imputed if best_model_name in ['Random Forest', 'Logistic Regression'] else X_test
    test_probs = best_model.predict_proba(X_test_used)[:, 1]
    
    # Sub 1: F1 Optimized
    test_preds_f1 = (test_probs >= best_f1_thresh).astype(int)
    sub_f1 = pd.DataFrame({'call_id': test_ids, 'has_ticket': test_preds_f1})
    sub_f1.to_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/submission_real_f1.csv', index=False)
    
    # Sub 2: Recall Optimized
    test_preds_rec = (test_probs >= best_recall_thresh).astype(int)
    sub_rec = pd.DataFrame({'call_id': test_ids, 'has_ticket': test_preds_rec})
    sub_rec.to_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/submission_real_recall.csv', index=False)
    
    print(f"\nSubmissions generated:")
    print(f"- submission_real_f1.csv (Flags: {test_preds_f1.sum()})")
    print(f"- submission_real_recall.csv (Flags: {test_preds_rec.sum()})")

if __name__ == "__main__":
    main()
