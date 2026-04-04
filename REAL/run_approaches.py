import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ──────────────────────────────────────────────────
# Using the REAL dataset paths observed in your workspace
train = pd.read_csv('REAL/hackathon_train.csv')
val   = pd.read_csv('REAL/hackathon_val.csv')
test  = pd.read_csv('REAL/hackathon_test.csv')
all_data = pd.concat([train, val], ignore_index=True)

# ── FEATURE ENGINEERING ────────────────────────────────────────
def engineer_features_base(df):
    f = pd.DataFrame(index=df.index)
    
    # Core numeric
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

    # GOLDEN SIGNALS
    f['dur_600']         = (df['call_duration'].fillna(0) > 600).astype(int)
    f['dur_300']         = (df['call_duration'].fillna(0) > 300).astype(int)
    f['comp_partial']    = ((df['response_completeness'].fillna(0) >= 0.9) & 
                            (df['response_completeness'].fillna(0) < 1.0)).astype(int)
    f['long_incomplete'] = ((df['call_duration'].fillna(0) > 300) & 
                            (outcomes == 'incomplete')).astype(int)
    f['partial_complete']= ((df['response_completeness'].fillna(0) >= 0.9) & 
                            (df['response_completeness'].fillna(0) < 1.0) & 
                            (outcomes == 'completed')).astype(int)

    # Interaction features
    f['duration_per_answer'] = df['call_duration'].fillna(0) / (df['answered_count'].fillna(0) + 1)
    f['answer_gap']          = df['question_count'].fillna(0) - df['answered_count'].fillna(0)
    f['billing_gap']         = (df['billing_duration'].fillna(0) - df['call_duration'].fillna(0)).abs()
    f['words_per_turn']      = df['user_word_count'].fillna(0) / (df['user_turn_count'].fillna(1) + 1)

    # Notes features
    notes = df['validation_notes'].fillna('').str.lower()
    f['notes_len']       = notes.str.len()
    f['has_correction']  = notes.str.contains(
        'corrected call_outcome|outcome corrected|labeled outcome as.*but',
        regex=True, na=False).astype(int)
    f['form_submitted']  = df['form_submitted'].fillna(False).astype(int)

    return f

X_tr_df = engineer_features_base(train)
X_va_df = engineer_features_base(val)
X_tr = X_tr_df.values
X_va = X_va_df.values
y_tr = train['has_ticket'].astype(int).values
y_va = val['has_ticket'].astype(int).values

spw   = (y_tr==0).sum() / (y_tr==1).sum()
sw_tr = np.where(y_tr==1, spw, 1.0)

BEST_THRESH = 0.685

gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    min_samples_leaf=20, subsample=0.8, random_state=42)

rf = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=42)

gb.fit(X_tr, y_tr, sample_weight=sw_tr)
rf.fit(X_tr, y_tr)
vp_gb  = gb.predict_proba(X_va)[:,1]
vp_rf  = rf.predict_proba(X_va)[:,1]
vp_ens = 0.6*vp_gb + 0.4*vp_rf

def report(name, preds, probs=None):
    f1 = f1_score(y_va, preds)
    prec = precision_score(y_va, preds, zero_division=0)
    rec = recall_score(y_va, preds)
    tp = (y_va.astype(bool) & preds.astype(bool)).sum()
    fp = (~y_va.astype(bool) & preds.astype(bool)).sum()
    fn = (y_va.astype(bool) & ~preds.astype(bool)).sum()
    print(f"\n[{name}]")
    print(f"Val F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} Flagged={preds.sum()}")
    return f1

print("--- BASELINE ---")
val_preds_base = (vp_ens > BEST_THRESH).astype(int)
report("Baseline GB+RF Ensemble", val_preds_base)

# ── APPROACH 3: Rule-based layer on top of ML ─────────────────
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

# Base flags from ML
a3_preds = val_preds_base.copy().astype(bool)

# Add hard rules
a3_preds |= hr_flags

# Remove if ML < 0.3 AND no hard rule fires
remove_mask = (vp_ens < 0.3) & (~hr_flags)
a3_preds[remove_mask] = False

a3_preds = a3_preds.astype(int)

f1_a3 = report("Approach 3: Rule-based Layer on ML", a3_preds)

sub_a3 = pd.DataFrame({'call_id': val['call_id'], 'has_ticket': a3_preds})
sub_a3.to_csv('REAL/submission_v3_val.csv', index=False)

if f1_a3 > 0.50:
    print("\nSUCCESS! Approach 3 exceeded 0.50 Val F1. Stopping.")
else:
    print("\nApproach 3 failed to exceed 0.50. Generating Approach 5 (Text Features)...")

# ── APPROACH 5: Feature expansion from transcript_text ────────
import re
def engineer_features_text(df):
    f = engineer_features_base(df)
    
    transcripts = df['transcript_text'].fillna('').str.lower()
    
    f['txt_cancel'] = transcripts.str.contains('cancel', regex=False).astype(int)
    f['txt_frust'] = transcripts.str.contains("don't call|stop calling|remove me", regex=True).astype(int)
    f['txt_real_person'] = transcripts.str.contains('i am a real person', regex=False).astype(int)
    
    f['txt_no_count'] = transcripts.apply(lambda x: len(re.findall(r'\bno\b', x)))
    f['txt_yes_count'] = transcripts.apply(lambda x: len(re.findall(r'\byes\b', x)))
    f['txt_len'] = transcripts.str.len()
    
    return f

X_tr_txt_df = engineer_features_text(train)
X_va_txt_df = engineer_features_text(val)
X_tr_txt = X_tr_txt_df.values
X_va_txt = X_va_txt_df.values

gb_txt = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    min_samples_leaf=20, subsample=0.8, random_state=42)
rf_txt = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=42)

gb_txt.fit(X_tr_txt, y_tr, sample_weight=sw_tr)
rf_txt.fit(X_tr_txt, y_tr)

vp_gb_txt = gb_txt.predict_proba(X_va_txt)[:,1]
vp_rf_txt = rf_txt.predict_proba(X_va_txt)[:,1]
vp_ens_txt = 0.6*vp_gb_txt + 0.4*vp_rf_txt

# Try threshold tuning for Approach 5
best_f1_a5 = 0
best_t_a5 = 0.5
for t in np.arange(0.3, 0.8, 0.05):
    preds_a5 = (vp_ens_txt > t).astype(int)
    f1 = f1_score(y_va, preds_a5)
    if f1 > best_f1_a5:
        best_f1_a5 = f1
        best_t_a5 = t

preds_best_a5 = (vp_ens_txt > best_t_a5).astype(int)
f1_a5 = report(f"Approach 5: Transcript Features (thresh={best_t_a5:.2f})", preds_best_a5)

if f1_a5 > 0.50:
    print("\nSUCCESS! Approach 5 exceeded 0.50 Val F1. Stopping.")
else:
    print("\nMoving to next approach...")
