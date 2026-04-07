import pandas as pd
import sys
sys.path.append('REAL')
import run_approach8 as app8

val = app8.val
y_va = app8.y_va
preds_val = app8.preds_val

fn_mask = (y_va == 1) & (preds_val == 0)
fp_mask = (y_va == 0) & (preds_val == 1)

fns = val[fn_mask]
fps = val[fp_mask]

print("--- FALSE NEGATIVES (True flags we MISSED) ---")
for idx, row in fns.iterrows():
    print(f"[{row['outcome'][:8]}] Dur={row['call_duration']} | Ans={row['answered_count']} | {str(row['validation_notes'])[:100]} | {str(row['transcript_text'])[:50]}")

print("\n--- FALSE POSITIVES (Fake flags we WRONGLY CAUGHT) ---")
for idx, row in fps.iterrows():
    print(f"[{row['outcome'][:8]}] Dur={row['call_duration']} | Ans={row['answered_count']} | {str(row['validation_notes'])[:100]} | {str(row['transcript_text'])[:50]}")
