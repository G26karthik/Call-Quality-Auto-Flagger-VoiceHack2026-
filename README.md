# CareCaller Hackathon 2026 — Call Quality Auto-Flagger

Binary classification system that predicts which AI-powered healthcare calls require human review (`has_ticket`).

**Live Demo:** [https://g26karthik.github.io/Call-Quality-Auto-Flagger-VoiceHack2026-/](https://g26karthik.github.io/Call-Quality-Auto-Flagger-VoiceHack2026-/)

> Run the full prediction engine in your browser — loads CSVs, applies all rules, and computes F1/precision/recall in real time.

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

- **0 false positives** across all labeled splits
- **18/159** test calls flagged (11.3%), consistent with training base rate
- **4th on public leaderboard** with perfect score

## How to Run

**Prerequisites:** Python 3.9+, pandas, scikit-learn

```bash
pip install pandas scikit-learn
python predict.py
```

This will:
- Evaluate on train and validation splits (prints F1, precision, recall)
- Generate `submission.csv` with test set predictions

Or try the **[live browser demo](https://g26karthik.github.io/Call-Quality-Auto-Flagger-VoiceHack2026-/)** — no Python required.

## File Structure

```
├── predict.py          # Rule engine, evaluation, and submission generation
├── submission.csv      # Test predictions (call_id, has_ticket as 0/1)
├── index.html          # GitHub Pages site with live JS-ported rule engine
├── README.md
├── PITCH_DECK.md       # 5-slide pitch deck blueprint
└── Datasets/
    ├── DATA_DICTIONARY.md
    ├── CareCaller_Hackathon_2026.pdf
    ├── csv/
    │   ├── hackathon_train.csv   (689 rows, 53 cols)
    │   ├── hackathon_val.csv     (144 rows)
    │   └── hackathon_test.csv    (159 rows)
    ├── json/
    └── parquet/
```

## Tech Stack

- **Python** — core runtime
- **pandas** — data loading and manipulation
- **scikit-learn** — evaluation metrics (F1, precision, recall)
- **re / json** (stdlib) — regex pattern matching and transcript parsing
- **JavaScript (PapaParse)** — browser-based live demo
