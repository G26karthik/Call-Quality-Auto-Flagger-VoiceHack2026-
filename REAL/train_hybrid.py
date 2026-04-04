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

def apply_layer1(df):
    flags = []
    for _, row in df.iterrows():
        outcome = str(row['outcome']).lower()
        call_duration = row['call_duration']
        resp_comp = row['response_completeness']
        notes = str(row.get('validation_notes', '')).lower()
        if pd.isna(notes) or notes == 'nan':
            notes = ''
            
        flag = 0
        if outcome in ['unknown', 'cancelled', 'voicemail']:
            flag = 1
        elif call_duration > 300 and resp_comp < 0.5:
            flag = 1
        elif 'labeled outcome as' in notes and 'but' in notes:
            flag = 1
        elif outcome == 'completed' and resp_comp < 0.95:
            flag = 1
        elif outcome == 'wrong_number':
            flag = 1
        elif call_duration == 0:
            flag = 1
            
        flags.append(flag)
    return np.array(flags)

def main():
    train, val, test = load_data()
    
    # Layer 1 application
    train['layer1_flag'] = apply_layer1(train)
    val['layer1_flag'] = apply_layer1(val)
    test['layer1_flag'] = apply_layer1(test)
    
    # Layer 1 Evaluation on Val
    y_val = val['has_ticket'].fillna(0).astype(int)
    val_l1_preds = val['layer1_flag']
    
    print("--- Layer 1 (Hard Rules) Only on Val ---")
    f1_l1 = f1_score(y_val, val_l1_preds)
    rec_l1 = recall_score(y_val, val_l1_preds)
    prec_l1 = precision_score(y_val, val_l1_preds, zero_division=0)
    print(f"F1: {f1_l1:.4f} | Recall: {rec_l1:.4f} | Precision: {prec_l1:.4f}")
    print(f"Number Flagged: {val_l1_preds.sum()}")
    print("-" * 40)
    
    # Layer 2 Preparation
    # Encode label 'outcome'
    le = LabelEncoder()
    # Combine to fit encoder
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
    
    lgb = LGBMClassifier(scale_pos_weight=68, random_state=42, verbose=-1)
    lgb.fit(X_train, y_train)
    
    # Layer 2 Evaluation on Val
    val_probs = lgb.predict_proba(X_val)[:, 1]
    val_l2_preds = (val_probs > 0.15).astype(int)
    
    # Layer 3 Combined on Val
    val_combined_preds = np.maximum(val_l1_preds, val_l2_preds)
    
    print("\n--- Layer 3 (Combined L1 + L2) on Val ---")
    f1_comb = f1_score(y_val, val_combined_preds)
    rec_comb = recall_score(y_val, val_combined_preds)
    prec_comb = precision_score(y_val, val_combined_preds, zero_division=0)
    print(f"F1: {f1_comb:.4f} | Recall: {rec_comb:.4f} | Precision: {prec_comb:.4f}")
    print(f"Number Flagged: {val_combined_preds.sum()}")
    print("-" * 40)
    
    # Generate Submissions on Test
    X_test = test[features]
    test_probs = lgb.predict_proba(X_test)[:, 1]
    test_l2_preds = (test_probs > 0.15).astype(int)
    
    test_combined_preds = np.maximum(test['layer1_flag'], test_l2_preds)
    
    sub = pd.DataFrame({
        'call_id': test['call_id'],
        'has_ticket': test_combined_preds
    })
    sub.to_csv('c:/Users/saita/OneDrive/Desktop/Projects/Caller.ai/REAL/submission_combined.csv', index=False)
    print(f"\nSaved submission_combined.csv with {test_combined_preds.sum()} flagged tickets.")

if __name__ == "__main__":
    main()
