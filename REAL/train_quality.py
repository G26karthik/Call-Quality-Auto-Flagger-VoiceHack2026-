import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from warnings import filterwarnings

filterwarnings("ignore")

def load_data():
    train = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_train.csv")
    val = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_val.csv")
    test = pd.read_csv("c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/hackathon_test.csv")
    return train, val, test

def get_signals_and_flags(df, ml_probs):
    all_signals = []
    flags = []
    for i, row in df.iterrows():
        signals = []
        outcome = str(row['outcome']).lower()
        call_duration = row['call_duration']
        response_completeness = row['response_completeness']
        pipeline_mismatch_count = row['pipeline_mismatch_count']
        notes = str(row.get('validation_notes', '')).lower()
        ml_proba = ml_probs[i]
        
        if call_duration > 200: signals.append('long_call')
        if outcome == 'completed' and response_completeness < 0.99: signals.append('incomplete_completed')
        if pipeline_mismatch_count > 2: signals.append('pipeline_mismatch')
        if outcome == 'wrong_number': signals.append('wrong_number')
        if outcome in ['unknown', 'cancelled']: signals.append('odd_outcome')
        if 'labeled outcome as' in notes and 'but' in notes: signals.append('outcome_mismatch')
        if call_duration == 0: signals.append('zero_duration')
        if ml_proba > 0.4: signals.append('ml_flag')
        
        all_signals.append(signals)
        flags.append(1 if len(signals) >= 2 else 0)
        
    return np.array(flags), all_signals

def main():
    train, val, test = load_data()
    
    # ML Layer Training
    le = LabelEncoder()
    train_val_test = pd.concat([train, val, test]).reset_index(drop=True)
    le.fit(train_val_test['outcome'].astype(str))
    
    for df in [train, val, test]:
        df['outcome_encoded'] = le.transform(df['outcome'].astype(str))
        df['duration_per_answer'] = df['call_duration'] / (df['answered_count'] + 1)
        
    features = [
        'call_duration', 'answered_count', 'response_completeness',
        'outcome_encoded', 'interruption_count', 'pipeline_mismatch_count',
        'attempt_number', 'duration_per_answer'
    ]
    
    X_train = train[features]
    y_train = train['has_ticket'].fillna(0).astype(int)
    X_val = val[features]
    y_val = val['has_ticket'].fillna(0).astype(int)
    
    lgb = LGBMClassifier(scale_pos_weight=68, random_state=42, verbose=-1)
    lgb.fit(X_train, y_train)
    
    val_probs = lgb.predict_proba(X_val)[:, 1]
    
    val_flags, val_signals_list = get_signals_and_flags(val, val_probs)
    
    print("--- Conservative High-Precision Evaluation on Val ---")
    f1 = f1_score(y_val, val_flags)
    rec = recall_score(y_val, val_flags)
    prec = precision_score(y_val, val_flags, zero_division=0)
    print(f"F1: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")
    print(f"Number Flagged: {sum(val_flags)}")
    print("-" * 40)
    
    print("\nFlagged Calls in Validation Set:")
    
    flagged_count = 0
    for i, flag in enumerate(val_flags):
        if flag == 1:
            call_id = val.iloc[i]['call_id']
            has_ticket = y_val.iloc[i]
            print(f"Call ID: {call_id} | Actual Ticket: {has_ticket} | Signals: {val_signals_list[i]}")
            flagged_count += 1
            
    # Generate Submission on Test
    X_test = test[features]
    test_probs = lgb.predict_proba(X_test)[:, 1]
    test_flags, test_signals_list = get_signals_and_flags(test, test_probs)
    
    sub = pd.DataFrame({
        'call_id': test['call_id'],
        'has_ticket': test_flags
    })
    sub.to_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/submission_quality.csv', index=False)
    print(f"\nSaved submission_quality.csv with {sum(test_flags)} flagged tickets.")

if __name__ == "__main__":
    main()
