import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('REAL/hackathon_train.csv')
val   = pd.read_csv('REAL/hackathon_val.csv')
test  = pd.read_csv('REAL/hackathon_test.csv')

def engineer_features_xgb(df):
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

    # Outcome encoding
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

    # Added features for XGB
    f['log_call_duration'] = np.log1p(df['call_duration'].fillna(0))
    f['log_answered_count'] = np.log1p(df['answered_count'].fillna(0))
    f['duration_sq'] = df['call_duration'].fillna(0)**2
    f['completeness_sq'] = df['response_completeness'].fillna(0)**2

    return f

X_tr_xgb = engineer_features_xgb(train).values
X_va_xgb = engineer_features_xgb(val).values
y_tr = train['has_ticket'].astype(int).values
y_va = val['has_ticket'].astype(int).values

clf = xgb.XGBClassifier(
    scale_pos_weight=67,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=500,
    early_stopping_rounds=50,
    eval_metric='logloss',
    random_state=42
)

clf.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

best_f1_a1 = 0
best_t_a1 = 0.5
vp_xgb = clf.predict_proba(X_va_xgb)[:,1]

for t in np.arange(0.3, 0.95, 0.05):
    preds_a1 = (vp_xgb > t).astype(int)
    f1 = f1_score(y_va, preds_a1)
    if f1 > best_f1_a1:
        best_f1_a1 = f1
        best_t_a1 = t

preds_best_a1 = (vp_xgb > best_t_a1).astype(int)
f1 = f1_score(y_va, preds_best_a1)
prec = precision_score(y_va, preds_best_a1, zero_division=0)
rec = recall_score(y_va, preds_best_a1)

tp = (y_va.astype(bool) & preds_best_a1.astype(bool)).sum()
fp = (~y_va.astype(bool) & preds_best_a1.astype(bool)).sum()
fn = (y_va.astype(bool) & ~preds_best_a1.astype(bool)).sum()

print("--- APPROACH 1: Better ML with XGBoost ---")
print(f"Val F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} (thresh={best_t_a1:.2f})")
print(f"TP={tp} FP={fp} FN={fn} Flagged={preds_best_a1.sum()}")

sub_a1 = pd.DataFrame({'call_id': val['call_id'], 'has_ticket': preds_best_a1})
sub_a1.to_csv('REAL/submission_v1_val.csv', index=False)

if f1 > 0.50:
    print("SUCCESS! Approach 1 exceeded 0.50 Val F1.")
