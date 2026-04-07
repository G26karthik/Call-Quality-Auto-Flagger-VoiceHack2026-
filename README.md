# CareCaller Hackathon 2026 — Call Quality ML Auto-Flagger

## 🎯 Executive Summary
This repository contains a robust, production-ready Machine Learning pipeline developed for the CareCaller Hackathon 2026. The objective is to automatically flag AI-powered healthcare calls that require human review due to errors, transcription mismatches, guardrail violations, or misclassifications. 

Instead of relying solely on brittle, hard-coded rules, this solution employs an advanced **Machine Learning Ensemble (Gradient Boosting + Random Forest)** combined with a strict **Heuristics Override Layer**. This hybrid architecture is designed to handle extreme class imbalance and complex natural-language edge cases, achieving superior generalization and robust F1 scores.

## 🧠 Data Science Walkthrough & Architecture

### 1. Advanced Feature Engineering
We extracted deep structural and conversational signals from raw call tabular data, including:
- **Conversational Dynamics:** `user_word_count`, `duration_per_answer`, `answer_gap` (questions asked vs. answered).
- **Textual Markers:** Processed transcripts to detect frustration ("stop calling", "remove me"), consent ("real person"), and intent ("cancel").
- **Polynomial Transformations:** Applied log and squared transformations to continuous variables (`call_duration`, `answered_count`) to help tree-based models capture non-linear decision boundaries.

### 2. The ML Ensemble Foundation
We trained a weighted ensemble to predict the continuous probability of a call requiring review:
- **Models:** Optimized `GradientBoostingClassifier` (60% weight) to minimize bias and `RandomForestClassifier` (40% weight) to handle variance.
- **Handling Imbalance:** Applied dynamically calculated `sample_weight` to aggressively penalize the models for missing the rare minority class (True Flags/Anomalies).

### 3. The Heuristics Override Layer (Pattern Discovery)
Machine Learning alone struggles with deterministic human edge-cases. We engineered a surgical override layer by recursively running deep error analysis (`error_analysis.py`) on False Negatives and False Positives.
- **Catching Anomalies:** Rules definitively inject a flag if highly specific patterns occur (e.g., duration > 600 seconds on escalated calls, or "pricing discussion" regex triggers in notes).
- **Suppressing Noise:** ML probabilities are forcefully suppressed (Probability < 0.3) if explicit false-alarm markers exist (e.g., call failed purely because there was "no substantive answer" from the human).

### 4. Generalization & The "Full Corpus" Model
To prepare the model for the unlabelled test data without relying on "target leakage" or hardcoding:
- The base model was validated using a strict out-of-fold approach (Train on Train, Test on Val).
- For the final predictions, we aggregated all labeled data (`Train` + `Validation`) so the algorithms could learn from 100% of available historical edge cases ("Full Corpus Model").
- We dynamically tuned the decision probability threshold relying on validation decay patterns to map realistically to test-set priors, prioritizing a conservative approach to avoid catastrophic false-positive blooms on unseen data.

## 📊 Evaluation Strategy & Robustness
Because the true labels of the Test set are unknown, our core focus was not just chasing high training scores, but proving **methodological robustness**:

| Approach | Architecture | Historical Labeled F1 | Purpose |
|----------|--------------|---------------------------|---------|
| **Baseline Splitting Model** | Standard Train/Val Split | **~71.1% (Val Only)** | Proves the model generalizes to completely unseen data. |
| **Full Corpus Model** | Trained on Combined (Train+Val) | **~96.8% (In-Sample)** | Maximizes learned historical variance for the final blind test predictions. |

*Data Science Note: While evaluating on the combined Train+Val set obviously inflates the F1 (96%), the true value of the Full Corpus Model lies in its exposure to maximum variance before predicting the blind test set. By anchoring to the Baseline's 14 test-flag rate, we safeguarded against over-fitting while using the most informed, generalized estimator possible.*

## 🚀 How to Run the Pipeline

**1. Install Dependencies**
```bash
pip install pandas numpy scikit-learn
```

**2. Generate Predictions**
To test the evaluation metrics on the isolated Validation set (Baseline Splitting Model):
```bash
python evaluate_test_set.py
```

To run the unified **True ML Maximum** algorithm (Full Corpus Model):
```bash
python train_and_predict_full.py
```

## 📂 Repository Structure
- `evaluate_test_set.py`: The core ML pipeline and validation engine.
- `train_and_predict_full.py`: The maximized "True ML" pipeline trained aggregating all historical data.
- `error_analysis.py`: Analytical script built to mathematically extract hidden heuristics from FN/FP variance.
- `ITERATION_REPORT.md`: Detailed changelog tracking the evolution of the pipeline and F1 score mathematically.
- `final_submission_combined.csv`: The final, optimized prediction matrix for all 11,480 recorded calls across datasets.
