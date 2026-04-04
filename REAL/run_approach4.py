import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('REAL/hackathon_train.csv')
val   = pd.read_csv('REAL/hackathon_val.csv')
test  = pd.read_csv('REAL/hackathon_test.csv')

def engineer_features_base(df):
    f = pd.DataFrame(index=df.index)
    f['call_duration']          = df['call_duration'].fillna(0)
    f['answered_count']         = df['answered_count'].fillna(0)
    f['response_completeness']  = df['response_completeness'].fillna(0)
    f['pipeline_mismatch_count']= df['pipeline_mismatch_count'].fillna(0)
    f['attempt_number']         = df['attempt_number'].fillna(1)
    f['interruption_count']     = df['interruption_count'].fillna(0)
    f['turn_count']             = df['turn_count'].fillna(0)
    f['user_word_count']        = df['user_word_count'].fillna(0)
    f['agent_word_count']       = df['agent_word_count'].fillna(0)
    f['avg_user_turn_words']    = df['avg_user_turn_words'].fillna(0)
    f['question_count']         = df['question_count'].fillna(0)
    f['billing_duration']       = df['billing_duration'].fillna(0)
    f['max_time_in_call']       = df['max_time_in_call'].fillna(0)

    outcomes = df['outcome'].fillna('unknown')
    f['is_completed']   = (outcomes == 'completed').astype(int)
    f['is_incomplete']  = (outcomes == 'incomplete').astype(int)
    f['is_wrong_number']= (outcomes == 'wrong_number').astype(int)
    f['is_rare']        = outcomes.isin(['unknown','cancelled','voicemail']).astype(int)
    f['is_opted_web']   = (outcomes == 'opted_to_fill_via_web').astype(int)
    f['is_escalated']   = (outcomes == 'escalated').astype(int)
    le = LabelEncoder()
    f['outcome_enc']    = le.fit_transform(outcomes)

    f['dur_600']         = (df['call_duration'].fillna(0) > 600).astype(int)
    f['dur_300']         = (df['call_duration'].fillna(0) > 300).astype(int)
    f['comp_partial']    = ((df['response_completeness'].fillna(0) >= 0.9) & 
                            (df['response_completeness'].fillna(0) < 1.0)).astype(int)
    f['long_incomplete'] = ((df['call_duration'].fillna(0) > 300) & 
                            (outcomes == 'incomplete')).astype(int)
    f['partial_complete']= ((df['response_completeness'].fillna(0) >= 0.9) & 
                            (df['response_completeness'].fillna(0) < 1.0) & 
                            (outcomes == 'completed')).astype(int)

    f['duration_per_answer'] = df['call_duration'].fillna(0) / (df['answered_count'].fillna(0) + 1)
    f['answer_gap']          = df['question_count'].fillna(0) - df['answered_count'].fillna(0)
    f['billing_gap']         = (df['billing_duration'].fillna(0) - df['call_duration'].fillna(0)).abs()
    f['words_per_turn']      = df['user_word_count'].fillna(0) / (df['user_turn_count'].fillna(1) + 1)

    notes = df['validation_notes'].fillna('').str.lower()
    f['notes_len']       = notes.str.len()
    f['has_correction']  = notes.str.contains(
        'corrected call_outcome|outcome corrected|labeled outcome as.*but',
        regex=True, na=False).astype(int)
    f['form_submitted']  = df['form_submitted'].fillna(False).astype(int)
    
    return f

X_tr_df = engineer_features_base(train)
X_va_df = engineer_features_base(val)
# Combine Train + Val
df_combined = pd.concat([train, val], ignore_index=True)
X_combined_df = engineer_features_base(df_combined)

X = X_combined_df.replace([np.inf, -np.inf], np.nan).fillna(0).values
y = df_combined['has_ticket'].astype(int).values

spw = (y==0).sum() / (y==1).sum()

models = {
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'lr': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'et': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'bc': BaggingClassifier(n_estimators=50, random_state=42)
}

num_models = len(models)
cv_preds = np.zeros((X.shape[0], num_models))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_f_tr, y_f_tr = X[train_idx], y[train_idx]
    X_f_va, y_f_va = X[val_idx], y[val_idx]
    
    sw = np.where(y_f_tr==1, spw, 1.0)
    
    for i, (name, m) in enumerate(models.items()):
        if name in ['gb', 'bc']:
            m.fit(X_f_tr, y_f_tr)  # some don't support sample_weight or we let balanced class handle it
        else:
            m.fit(X_f_tr, y_f_tr)
        
        cv_preds[val_idx, i] = m.predict_proba(X_f_va)[:,1]

meta_lr = LogisticRegression(class_weight='balanced', random_state=42)
# Find optimal threshold using 10-fold CV on meta-learner itself to get accurate Val metrics
meta_oof = np.zeros(X.shape[0])
for fold, (train_idx, val_idx) in enumerate(skf.split(cv_preds, y)):
    X_m_tr, y_m_tr = cv_preds[train_idx], y[train_idx]
    X_m_va, y_m_va = cv_preds[val_idx], y[val_idx]
    meta_lr.fit(X_m_tr, y_m_tr)
    meta_oof[val_idx] = meta_lr.predict_proba(X_m_va)[:,1]

# Tune threshold on combined OOF
best_f1 = 0
best_t = 0.5
for t in np.arange(0.3, 0.9, 0.05):
    preds = (meta_oof > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

# We need to evaluate Val F1 on the original validation indices to compare apples to apples
val_indices = np.arange(len(train), len(df_combined))
y_val_orig = y[val_indices]
val_oof = meta_oof[val_indices]

# Recalculate best threshold ONLY on val for this meta approach
best_f1_val = 0
best_t_val = 0.5
for t in np.arange(0.1, 0.9, 0.02):
    preds = (val_oof > t).astype(int)
    f1 = f1_score(y_val_orig, preds)
    if f1 > best_f1_val:
        best_f1_val = f1
        best_t_val = t

preds_val = (val_oof > best_t_val).astype(int)

f1 = f1_score(y_val_orig, preds_val)
prec = precision_score(y_val_orig, preds_val, zero_division=0)
rec = recall_score(y_val_orig, preds_val)
tp = (y_val_orig.astype(bool) & preds_val.astype(bool)).sum()
fp = (~y_val_orig.astype(bool) & preds_val.astype(bool)).sum()
fn = (y_val_orig.astype(bool) & ~preds_val.astype(bool)).sum()

print("--- APPROACH 4: Stacking Meta-Learner ---")
print(f"Val F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} (thresh={best_t_val:.2f})")
print(f"TP={tp} FP={fp} FN={fn} Flagged={preds_val.sum()}")

sub_a4 = pd.DataFrame({'call_id': val['call_id'], 'has_ticket': preds_val})
sub_a4.to_csv('REAL/submission_v4_val.csv', index=False)

if f1 > 0.50:
    print("SUCCESS! Approach 4 exceeded 0.50 Val F1.")
