import pandas as pd
train_df = pd.read_csv('../Datasets/csv/hackathon_train.csv')
val_df = pd.read_csv('../Datasets/csv/hackathon_val.csv')

# Keywords that appear in ticket validation notes
tickets = train_df[train_df['has_ticket'] == True]
non_tickets = train_df[train_df['has_ticket'] == False]

keywords = ['fabricated', 'erroneously', 'dosage guidance', 'guidance', 'miscategorization', 
            'opted_out', 'wrong_number', 'normalization', 'medical', 'advice', 'STT', 
            'speech-to-text', 'audio', 'glitch', 'mismatch', 'incorrect', 'error']

print('Keyword frequency in TICKETS vs NON-TICKETS:')
print('-' * 60)
for kw in keywords:
    in_tickets = tickets['validation_notes'].str.lower().str.contains(kw.lower(), na=False).sum()
    in_non = non_tickets['validation_notes'].str.lower().str.contains(kw.lower(), na=False).sum()
    if in_tickets > 0 or in_non > 0:
        print(f'{kw:20} | Tickets: {in_tickets:3} | Non: {in_non:3} | Ratio: {in_tickets/(in_non+0.1):.2f}')

print()
print('=== VAL SET tickets with whisper_mismatch=0 ===')
val_tickets_0 = val_df[(val_df['has_ticket'] == True) & (val_df['whisper_mismatch_count'] == 0)]
print(f'Count: {len(val_tickets_0)}')
for i, row in val_tickets_0.iterrows():
    print(f"\nOutcome: {row['outcome']}")
    print(f"Notes: {row['validation_notes'][:300]}")
