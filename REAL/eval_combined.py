import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import run_approach8 as app8

# Prepare train set predictions (Validation & Test predictions are handled inside run_approach8 itself)
vp_tr_ens = app8.get_ensemble_probs(app8.X_tr)
hr_flags_tr = app8.hard_rules_layer(app8.train).values

# 1. Base ML threshold
preds_tr = (vp_tr_ens > app8.best_t).astype(bool)

# 2. Add Hard Rules override
preds_tr |= hr_flags_tr

# 3. Apply explicit outlier / false-positive removals
remove_mask_tr = (vp_tr_ens < 0.3) & (~hr_flags_tr)
remove_mask_tr |= (app8.train['call_duration'].fillna(0) > 600) & (app8.train['answered_count'].fillna(0) == 1)
preds_tr[remove_mask_tr] = False

preds_tr = preds_tr.astype(int)

# Combine datasets (Train + Val)
y_comb = np.concatenate([app8.y_tr, app8.y_va])
preds_comb = np.concatenate([preds_tr, app8.preds_val])

print("\n\n====== COMPREHENSIVE COMBINED RESULTS (TRAIN + VAL) ======")
print(f"Combined TP = {(y_comb.astype(bool) & preds_comb.astype(bool)).sum()}")
print(f"Combined FP = {(~y_comb.astype(bool) & preds_comb.astype(bool)).sum()}")
print(f"Combined FN = {(y_comb.astype(bool) & ~preds_comb.astype(bool)).sum()}")
print(f"Combined F1 Score:   {f1_score(y_comb, preds_comb):.4f}")
print(f"Combined Precision:  {precision_score(y_comb, preds_comb, zero_division=0):.4f}")
print(f"Combined Recall:     {recall_score(y_comb, preds_comb):.4f}")

print("\n====== TEST SET (UNLABELED) PREDICTIONS ======")
print(f"Total rows in Test Set: {len(app8.test)}")
print(f"Total calls flagged as 'has_ticket' in Test Set: {app8.preds_test.sum()}")
print("Note: True test accuracy cannot be known until 'has_ticket' ground truth is provided by the hackathon evaluation platform.")
