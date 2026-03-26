# CareCaller Hackathon 2026 — Call Quality Auto-Flagger

Binary classification system that predicts which AI-powered healthcare calls require human review (`has_ticket`).

## Problem

CareCaller uses AI agents to conduct medication refill check-in calls. Some calls produce errors — STT mismatches, guardrail violations, miscategorized outcomes, or fabricated responses — and need human review. The goal is to automatically flag these calls with high precision and recall.

## Approach

Pure rule-based classifier operating on four signal layers:

1. **Validation notes pattern matching** — 22 regex patterns across 6 categories:
   - STT/audio errors (weight discrepancies, transcription errors)
   - Medical advice violations (dosage guidance, guardrail breaches)
   - Outcome misclassification (miscategorized calls, wrong-number identity issues)
   - Data contradictions (inconsistencies between recorded and actual responses)
   - Skipped questions (fabricated responses for unasked questions)
   - Opted-out miscategorization (patient expressed interest but classified as opted out)

2. **Whisper mismatch detection** — Any call with `whisper_mismatch_count > 0` is flagged as an audio issue.

3. **Structural signals** — Completed calls with incomplete responses (`0 < response_completeness < 1.0`) and incomplete calls meeting specific engagement thresholds.

4. **Transcript-level contradiction detection** — Parses `transcript_text` and `responses_json` to identify cases where recorded answers contradict what the patient actually said (e.g., side effects recorded as "Yes, mild nausea" when the patient said "No").

## Results

| Split | F1 | Precision | Recall | TP | FP | FN |
|-------|------|-----------|--------|----|----|-----|
| Train | 1.00 | 1.00 | 1.00 | 59/59 | 0 | 0 |
| Val | 1.00 | 1.00 | 1.00 | 11/11 | 0 | 0 |
| Public LB | 1.00 | — | — | — | — | — |

## How to Run

**Prerequisites:** Python 3.9+, pandas, scikit-learn

```bash
pip install pandas scikit-learn
```

**Generate predictions:**

```bash
python predict.py
```

This will:
- Evaluate on train and validation splits (prints F1, precision, recall)
- Generate `submission.csv` with test set predictions

## File Structure

```
├── predict.py          # Rule engine, evaluation, and submission generation
├── submission.csv      # Test predictions (call_id, has_ticket)
├── README.md
└── Datasets/
    └── csv/
        ├── hackathon_train.csv
        ├── hackathon_val.csv
        └── hackathon_test.csv
```

## Tech Stack

- Python
- pandas
- scikit-learn (metrics)
