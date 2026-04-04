import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_train.csv')
val   = pd.read_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_val.csv')

def engineer_features_all(df):
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

    # Text features (App 5)
    import re
    transcripts = df['transcript_text'].fillna('').str.lower()
    f['txt_cancel'] = transcripts.str.contains('cancel', regex=False).astype(int)
    f['txt_frust'] = transcripts.str.contains("don't call|stop calling|remove me", regex=True).astype(int)
    f['txt_real_person'] = transcripts.str.contains('i am a real person', regex=False).astype(int)
    f['txt_no_count'] = transcripts.apply(lambda x: len(re.findall(r'\bno\b', x)))
    f['txt_yes_count'] = transcripts.apply(lambda x: len(re.findall(r'\byes\b', x)))
    f['txt_len'] = transcripts.str.len()

    # XGB features (App 1)
    f['log_call_duration'] = np.log1p(df['call_duration'].fillna(0))
    f['log_answered_count'] = np.log1p(df['answered_count'].fillna(0))
    f['duration_sq'] = df['call_duration'].fillna(0)**2
    f['completeness_sq'] = df['response_completeness'].fillna(0)**2

    return f

X_tr_all = engineer_features_all(train)
X_va_all = engineer_features_all(val)
X_tr = X_tr_all.values
X_va = X_va_all.values
y_tr = train['has_ticket'].astype(int).values
y_va = val['has_ticket'].astype(int).values

spw   = (y_tr==0).sum() / (y_tr==1).sum()
sw_tr = np.where(y_tr==1, spw, 1.0)

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_samples_leaf=20, subsample=0.8, random_state=42)
rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=42)

gb.fit(X_tr, y_tr, sample_weight=sw_tr)
rf.fit(X_tr, y_tr)

vp_gb  = gb.predict_proba(X_va)[:,1]
vp_rf  = rf.predict_proba(X_va)[:,1]
vp_ens = 0.6*vp_gb + 0.4*vp_rf

def hard_rules_layer(df):
    flags = pd.Series(False, index=df.index)
    notes = df['validation_notes'].fillna('').str.lower()
    outcomes = df['outcome'].fillna('unknown')
    
    flags |= outcomes.isin(['unknown','cancelled','voicemail'])
    flags |= ((df['response_completeness'].fillna(0) >= 0.9) &
              (df['response_completeness'].fillna(0) < 1.0) &
              (outcomes == 'completed'))
    flags |= (df['call_duration'].fillna(0) > 600)
    flags |= (df['call_duration'].fillna(0) == 0)
    flags |= (notes.str.contains('labeled outcome as', na=False) &
              notes.str.contains('but', na=False))
    return flags

hr_flags = hard_rules_layer(val).values

best_t_a6 = 0.72

preds_best_a6 = (vp_ens > best_t_a6).astype(bool)
preds_best_a6 |= hr_flags
preds_best_a6[(vp_ens < 0.3) & (~hr_flags)] = False
preds_best_a6 = preds_best_a6.astype(int)

# Analysis
fps = np.where((preds_best_a6 == 1) & (y_va == 0))[0]
fns = np.where((preds_best_a6 == 0) & (y_va == 1))[0]
tps = np.where((preds_best_a6 == 1) & (y_va == 1))[0]

print("--- FALSE POSITIVES (FPs) ---")
for idx in fps:
    row = val.iloc[idx]
    ml_prob = vp_ens[idx]
    hr_trig = hr_flags[idx]
    trig = "Hard Rule" if ((not (ml_prob > best_t_a6)) and hr_trig) else ("ML" if (ml_prob > best_t_a6 and not hr_trig) else "Both")
    notes = str(row['validation_notes'])[:200].replace('\n', ' ')
    print(f"ID: {row['call_id'][:8]} | Out: {row['outcome']} | Dur: {row['call_duration']} | Comp: {row['response_completeness']:.2f} | Ans: {row['answered_count']} | Mis: {row['pipeline_mismatch_count']} | ML_Prob: {ml_prob:.3f} | Trig: {trig} | Notes: {notes}")

print("\n--- FALSE NEGATIVES (FNs) ---")
for idx in fns:
    row = val.iloc[idx]
    ml_prob = vp_ens[idx]
    notes = str(row['validation_notes'])[:200].replace('\n', ' ')
    t_notes = str(row.get('ticket_initial_notes', ''))[:200].replace('\n', ' ')
    print(f"ID: {row['call_id'][:8]} | Out: {row['outcome']} | Dur: {row['call_duration']} | Comp: {row['response_completeness']:.2f} | Ans: {row['answered_count']} | Mis: {row['pipeline_mismatch_count']} | ML_Prob: {ml_prob:.3f} | T_Notes: {t_notes} | V_Notes: {notes}")
    
print("\n--- TRUE POSITIVES (TPs) ---")
for idx in tps:
    row = val.iloc[idx]
    ml_prob = vp_ens[idx]
    hr_trig = hr_flags[idx]
    trig = "Hard Rule" if ((not (ml_prob > best_t_a6)) and hr_trig) else ("ML" if (ml_prob > best_t_a6 and not hr_trig) else "Both")
    notes = str(row['validation_notes'])[:200].replace('\n', ' ')
    print(f"ID: {row['call_id'][:8]} | Out: {row['outcome']} | Dur: {row['call_duration']} | Comp: {row['response_completeness']:.2f} | Ans: {row['answered_count']} | Mis: {row['pipeline_mismatch_count']} | ML_Prob: {ml_prob:.3f} | Trig: {trig} | Notes: {notes}")
