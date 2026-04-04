# Production Readiness Checklist

## Current Production Pipeline

Primary runner: `REAL/train_real.py`

The pipeline now optimizes for hidden leaderboard scoring with a composite objective that balances:

- Flag quality (precision)
- Coverage (recall and useful flag volume)
- Stability (accuracy)

It also emits multiple profile-specific submissions in one run.

## Active Model Stack

- LightGBM
- XGBoost (CUDA enabled)
- Random Forest
- Logistic Regression
- CatBoost (automatically used when available)

## Run Commands

Fast production run:

```bash
cd REAL
python train_real.py
```

Diagnostic run with CV summary:

```bash
cd REAL
python train_real.py --with-cv
```

Choose a default submission profile explicitly:

```bash
python train_real.py --profile balanced
python train_real.py --profile recall
python train_real.py --profile quality
python train_real.py --profile volume
```

## Generated Outputs

- `REAL/submission_real.csv` (primary profile selected by `--profile`)
- `REAL/submission_real_balanced.csv`
- `REAL/submission_real_recall.csv`
- `REAL/submission_real_quality.csv`
- `REAL/submission_real_volume.csv`

## Profile Guidance

- `balanced`: best blend for hidden mixed scoring (recommended default)
- `recall`: higher ticket coverage, lower precision
- `quality`: cleaner flags, fewer total flags
- `volume`: maximum flags; use only if scoring strongly rewards coverage

## Diagnostic Signals Printed Per Run

- Holdout threshold sweep
- Profile winner table with F1, precision, recall, accuracy, flagged count
- Calibration diagnostic (approx ECE)
- Threshold stability around selected threshold (±0.02)
- CV summary when `--with-cv` is enabled

## Operational Recommendation

1. Submit `balanced` first for safest hidden-score performance.
2. If score underperforms and you suspect recall-heavy weighting, submit `recall`.
3. Use `quality` when false positives seem heavily penalized.
4. Keep `volume` as an exploration option, not default.
