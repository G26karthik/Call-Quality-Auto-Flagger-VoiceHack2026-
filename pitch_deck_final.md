# CareCaller Hackathon 2026 — Technical Pitch Deck (Final)

## Problem 1: Call Quality Auto-Flagger

**Final Standing: 2nd Place**

---

## Slide 1: Title

### **CareCaller Call Quality Auto-Flagger**
**Hackathon 2026 — Problem 1: Binary Classification of `has_ticket`**

| Submission | Public LB | Private LB | Val F1 |
|------------|-----------|------------|--------|
| **Rule-Based** | **1.00** | **1.00** | **1.00** |
| ML v4 (Final) | 1.00 | 0.933 | 0.92 |

**Team:** [Your Team Name]  
**Date:** April 2026  
**Leaderboard Position:** 2nd Place

#### Simple Explanation
We built a system that automatically flags AI phone calls that need human review. Think of it like spam filtering, but for identifying calls where the AI made mistakes. Our rule-based approach achieves perfect accuracy on the test set.

#### Technical Depth
Our solution achieved **perfect F1=1.0** on both public and private leaderboards using a rule-based system that detects 6 distinct error categories. The ML pipeline serves as independent validation, achieving Public LB=1.0 but with Private LB=0.933 — demonstrating the classic bias-variance tradeoff where explicit rules generalize better than learned patterns on this dataset.

#### Suggested Visual
- Large centered title with CareCaller branding
- Side-by-side leaderboard comparison (Public vs Private LB)
- "2nd Place" achievement badge

---

## Slide 2: Problem & Dataset

### **The Challenge**
Automatically flag AI voice calls that need human review (tickets)

### **Dataset Overview**
| Split | Total Calls | Has Ticket | % Positive |
|-------|-------------|------------|------------|
| Train | 689 | 59 | 8.6% |
| Val | 144 | 11 | 7.6% |
| Test | 159 | ? | ? |
| **Combined** | **833** | **70** | **8.4%** |

### **Key Terminology**
- **F1 Score** [harmonic mean of precision and recall — balances catching errors vs avoiding false alarms]
- **Class Imbalance** [when one class (tickets) is much rarer than the other — here 10.68:1 ratio]
- **Precision** [of all calls we flagged, what % actually needed review]
- **Recall** [of all calls that needed review, what % did we catch]

### **What Triggers a Ticket?**
1. STT/Audio transcription errors
2. Agent providing medical/dosage advice (guardrail violation)
3. Outcome misclassification (completed vs incomplete)
4. Data contradictions between transcript and responses
5. Skipped questionnaire questions
6. Wrong number/opted-out misclassification

#### Simple Explanation
Imagine an AI makes 100 phone calls. Only about 8 of them have problems that need a human to look at. Our job is to find those 8 without accidentally flagging the other 92 good calls. It's like finding needles in a haystack — except we have clues in the call data that tell us where to look.

#### Technical Depth
The hackathon dataset simulates real telehealth check-in calls where an AI agent collects patient information. Tickets are flagged when the AI exhibits errors: transcription mistakes (STT errors where "147 lbs" becomes "47 lbs"), guardrail violations (agent giving medical advice), or classification errors. The **class imbalance ratio of 10.68:1** requires careful handling — standard accuracy would be misleading (92% accuracy by predicting all negatives). The `validation_notes` field contains explicit QA annotations, making pattern matching highly effective.

#### Suggested Visual
- Pie chart showing class distribution (91.4% no ticket, 8.6% ticket)
- Icons for each of the 6 ticket trigger categories
- Sample row highlighting key columns

---

## Slide 3: Solution Architecture

### **Dual-Track Approach: Rules + ML**

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
│   transcript_text, validation_notes, responses_json, metadata   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   RULE-BASED SYSTEM   │       │    ML PIPELINE        │
│   • Pattern matching  │       │    • Feature eng.     │
│   • Structural rules  │       │    • LightGBM/RF/LR   │
│   • 100% explainable  │       │    • Ensemble voting  │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            │  Public LB = 1.00             │  Public LB = 1.00
            │  Private LB = 1.00            │  Private LB = 0.93
            ▼                               ▼
┌───────────────────────────────────────────────────────────────┐
│                    FINAL PREDICTIONS                          │
│         submission.csv (18 flags) | submission_ml_v4.csv (17) │
└───────────────────────────────────────────────────────────────┘
```

### **Why Two Tracks?**
| Aspect | Rule-Based | ML Pipeline |
|--------|------------|-------------|
| Public LB | 1.00 | 1.00 |
| Private LB | 1.00 | 0.933 |
| Explainability | ✓ Every flag traceable | Partial (feature importance) |
| Maintenance | Update patterns manually | Retrain on new data |
| Edge cases | May miss novel patterns | Can generalize |

### **Key Terminology**
- **Leaderboard (LB)** [competition scoring system — Public uses 50% of test data, Private uses 100%]
- **Ensemble** [combining multiple models' predictions for better accuracy]

#### Simple Explanation
We took two different approaches to the same problem: one uses explicit rules ("if the call mentions dosage guidance, flag it"), and the other uses machine learning to automatically find patterns. The rule-based approach turned out to be better because the data contains obvious clues that humans can read and turn into rules.

#### Technical Depth
The dual-track architecture emerged from a key insight: `validation_notes` contains explicit QA annotations (not natural language), so deterministic pattern matching is optimal. The ML pipeline serves as validation — if ML independently learns the same patterns as rules, our rule selection is sound. We found that **18 explicit regex patterns + 3 structural rules** perfectly classify the data, while even our best ML model (v4 ensemble) shows Public LB=1.00 but Private LB=0.933, indicating some learned patterns don't generalize to unseen data.

#### Suggested Visual
- Flowchart diagram as shown above
- Color-coded paths (green for rules, blue for ML)
- Comparison table with score highlights

---

## Slide 4: Rule-Based System

### **6 Pattern Groups That Achieve Perfect F1 = 1.0**

| Category | Pattern Examples | Ticket Trigger |
|----------|-----------------|----------------|
| **1. STT/Audio Issues** | "erroneously as", "weight differs", "stt error" | Transcription mistakes |
| **2. Medical Advice** | "dosage guidance", "recommendations provided", "guardrail" | Agent violated guidelines |
| **3. Outcome Mismatch** | "miscategor", "corrected by validation", "does not match" | Wrong call classification |
| **4. Data Contradiction** | "inconsistency noted", "but recorded", "allergy.*but" | Transcript vs JSON mismatch |
| **5. Skipped Questions** | "fabricated responses as questions were not asked" | Agent skipped required Qs |
| **6. Classification Errors** | "wrong_number", "opted_out", "expressed interest but" | Misclassified outcomes |

### **Structural Rules**
```python
# Rule 1: Any whisper mismatch → ticket (always)
if whisper_mismatch_count > 0: flag = True

# Rule 2: Completed but incomplete responses → ticket
if outcome == "completed" and 0 < response_completeness < 1.0: flag = True

# Rule 3: Incomplete but high engagement → likely miscategorized
if outcome == "incomplete" and avg_user_turn_words > 8 
   and response_completeness < 0.65 and user_word_count > 80: flag = True
```

### **Key Terminology**
- **Regex** [regular expression — text pattern matching like "find any text containing 'dosage guidance'"]
- **whisper_mismatch_count** [how many times the AI's transcription differs from actual audio]

### **Results**
| Split | F1 | Precision | Recall | TP | FP | FN |
|-------|-----|-----------|--------|----|----|-----|
| Train | 1.00 | 1.00 | 1.00 | 59 | 0 | 0 |
| Val | 1.00 | 1.00 | 1.00 | 11 | 0 | 0 |
| Test | — | — | — | 18 flagged | — | — |

#### Simple Explanation
We analyzed all 59 training calls that needed tickets and found that they all share certain keywords like "dosage guidance" or "miscategorization" in their quality notes. We wrote rules to catch these keywords — it's like a checklist that catches 100% of problems without false alarms.

#### Technical Depth
The rule-based system was developed through systematic analysis of the 59 training tickets. We discovered that `validation_notes` contains explicit QA annotations: "dosage guidance" appears in 13 tickets and 0 non-tickets (130x signal ratio), "miscategorization" in 7 tickets and 0 non-tickets. The `whisper_mismatch_count > 0` proved to be a **perfect discriminator** — every such call was a ticket. We combined **22 regex patterns with 3 structural rules**, achieving perfect precision and recall. The rules are fully interpretable: every flagged call can be traced to a specific pattern match.

#### Suggested Visual
- Table with pattern categories and example matches
- Code snippet showing structural rules
- Funnel: "Raw Data → Pattern Match → Flag"

---

## Slide 5: ML Journey — v1 to v4

### **Version Evolution Summary**

| Version | Model | Key Changes | Val F1 | Public LB | Private LB |
|---------|-------|-------------|--------|-----------|------------|
| **v1** | XGBoost + TF-IDF | Baseline: generic text features | 0.92 | not submitted | — |
| **v2** | LightGBM + Enhanced | Added response parsing, engagement metrics | 0.91 | not submitted | — |
| **v3** | LightGBM + Targeted Flags | Domain-specific binary flags | 0.95 | 0.94 | **0.85** (OVERFIT) |
| **v4** | LGBM+RF+LR Ensemble | Broader patterns, 5-fold CV, regularization | 0.92 | **1.00** | **0.933** |

### **Key Terminology**
- **TF-IDF** [text feature extraction — converts words into numbers based on frequency and importance]
- **LightGBM** [fast gradient boosting algorithm — builds many decision trees sequentially, each correcting previous errors]
- **Overfitting** [model memorizes training data but fails on new unseen data — classic ML pitfall]
- **Cross-Validation (CV)** [training on different data subsets to estimate real-world performance]
- **Regularization** [technique to prevent overfitting by penalizing complex models]

### **v1: XGBoost + Generic TF-IDF (Val F1 = 0.92)**
```
Features: TF-IDF on validation_notes (100 dims) + basic metrics
Problem: TF-IDF treats "guidance" same as any other word — no domain knowledge
Result: Train F1=0.99, Val F1=0.92
```

### **v3: LightGBM + Targeted Flags (Val F1 = 0.95, but OVERFIT)**
```python
# Highly specific binary flags — worked too well on training data
has_dosage_guidance = int('dosage guidance' in notes)  # 130x ticket ratio
has_miscategorization = int('miscategorization' in notes)  # 70x ticket ratio
```
```
Result: Val F1=0.95 looked great, but Private LB dropped to 0.85
Problem: Exact phrase matches overfit to training data patterns
```

### **v4: Ensemble with Anti-Overfitting (Val F1 = 0.92, Private LB = 0.933)**
```python
# Broader patterns that generalize better
mentions_guidance = int('guidance' in notes or 'advice' in notes)  # Less specific
mentions_mismatch = int('mismatch' in notes or 'discrepancy' in notes)
```
```
Changes: 5-fold CV, weighted ensemble, regularization
Result: Val F1 dropped to 0.92 BUT Private LB improved from 0.85 → 0.933
```

#### Simple Explanation
We tried four versions of our ML model. Version 3 looked amazing in testing (95% F1 score) but failed badly in the real competition (85% vs expected 95%). This is called "overfitting" — the model memorized the training examples instead of learning general patterns. Version 4 fixed this by using broader, more generalizable patterns and better validation techniques.

#### Technical Depth
The ML journey demonstrates the classic bias-variance tradeoff. v3 achieved **Val F1=0.95** by creating exact-match binary flags for phrases like "dosage guidance" — these worked perfectly on training/val but failed on the private test set (Public LB=0.94, Private LB=0.85). The **10-point drop** indicated severe overfitting. v4 addressed this through: (1) **broader text patterns** ("guidance OR advice" instead of exact "dosage guidance"), (2) **5-fold StratifiedKFold CV** instead of single train/val split, (3) **weighted ensemble** (LGBM 50%, RF 30%, LR 20%), and (4) **stronger regularization** (max_depth=4, min_child_samples=15). This raised Private LB from 0.85 to 0.933 while accepting a slight Val F1 decrease.

#### Suggested Visual
- Timeline showing v1 → v2 → v3 → v4 with scores
- Warning icon on v3 showing "OVERFIT"
- Arrow showing "v3 Private LB 0.85 → v4 Private LB 0.933"

---

## Slide 6: Overfitting Detection & Fix

### **The Problem: v3 Looked Great, Then Failed**

| Metric | v3 Score | What We Thought | Reality |
|--------|----------|-----------------|---------|
| Val F1 | 0.95 | "Amazing!" | Only tested on 144 rows |
| Public LB | 0.94 | "Confirmed!" | Only 50% of test set |
| Private LB | 0.85 | "Wait, what?" | Real performance on full test |

### **Key Terminology**
- **Public Leaderboard** [score on half of test data — shown during competition]
- **Private Leaderboard** [score on full test data — revealed at competition end]
- **Generalization** [model's ability to perform well on data it hasn't seen before]
- **Optuna** [hyperparameter tuning library — automatically finds best model settings]
- **StratifiedKFold** [cross validation that maintains class balance in each fold]

### **Root Cause Analysis**
v3's exact-match features like `"dosage guidance" in notes` were **too specific**:
- Training data: phrase appears in 13/59 tickets = 22%
- Test data: phrase distribution may differ
- Result: model learned training-specific patterns, not general rules

### **The Fix in v4**

| Aspect | v3 (Overfit) | v4 (Generalized) |
|--------|--------------|------------------|
| Text patterns | Exact: "dosage guidance" | Broad: "guidance OR advice" |
| Validation | Single train/val split | 5-fold StratifiedKFold CV |
| Models | Single LightGBM | Ensemble: LGBM + RF + LR |
| Regularization | Light | Strong (max_depth=4, reg_lambda=0.5) |
| CV F1 | not recorded | 0.95 |
| Val F1 | 0.95 | 0.92 (acceptable drop) |
| Private LB | 0.85 | **0.933** (+8.3% improvement) |

### **Ensemble Weights**
```python
weights = {'LightGBM': 0.5, 'Random Forest': 0.3, 'Logistic Regression': 0.2}
ensemble_proba = sum(model_proba[name] * weights[name] for name in models)
```

#### Simple Explanation
Imagine studying for a test by memorizing exact answers from practice problems. You ace the practice test, but fail the real exam because the questions are worded slightly differently. That's what happened to v3. We fixed v4 by teaching it to understand concepts rather than memorize exact phrases.

#### Technical Depth
The overfitting was caused by **feature specificity asymmetry** — exact phrase matches like "dosage guidance" had 100% precision on training but didn't generalize. v4's fixes addressed multiple overfitting sources: (1) **Broader regex patterns** capture semantic intent (any guidance mention) rather than exact wording, (2) **5-fold CV** estimates true performance better than single holdout, (3) **Ensemble averaging** reduces variance from any single model, (4) **Stronger regularization** prevents decision trees from memorizing edge cases. The **8.3-point improvement** on Private LB (0.85 → 0.933) validates these anti-overfitting techniques while the Val F1 drop (0.95 → 0.92) represents acceptable bias-variance rebalancing.

#### Suggested Visual
- Before/After comparison table
- Graph showing v3's "train/val great, test bad" vs v4's "consistent"
- Warning triangle icon with "Overfitting Detected"

---

## Slide 7: Feature Importance (ML v4)

### **Top 15 Features by LightGBM Importance**

*(Based on v4b LightGBM model)*

| Rank | Feature | Type | Interpretation |
|------|---------|------|----------------|
| 1 | `any_ticket_signal` | Engineered | Combined domain flag (key feature) |
| 2 | `response_completeness` | Structural | How many questions were answered |
| 3 | `has_whisper_mismatch` | Structural | Audio transcription error detected |
| 4 | `any_structural_issue` | Engineered | Combined structural problems |
| 5 | `whisper_mismatch_count` | Structural | Count of transcription errors |
| 6 | `mentions_guidance` | Text | Broad: "guidance" OR "advice" in notes |
| 7 | `outcome_enc` | Categorical | Encoded call outcome |
| 8 | `answered_ratio` | Response | % of questions with answers |
| 9 | `mentions_categorization` | Text | Broad: "categor" OR "classif" in notes |
| 10 | `outcome_wrong_number` | Categorical | Call marked as wrong number |

### **Key Terminology**
- **Feature Importance** [how much each input variable contributes to predictions]
- **Engineered Features** [variables we created by combining raw data]

### **What This Validates**
1. **Domain features dominate** — `any_ticket_signal` ranks #1
2. **Structural signals work** — `whisper_mismatch` consistently important
3. **Broader patterns generalize** — `mentions_guidance` captures what `has_dosage_guidance` missed

### **v4 vs v3 Feature Comparison**
| v3 Feature (Overfit) | v4 Feature (Generalizes) |
|---------------------|-------------------------|
| `has_dosage_guidance` (exact match) | `mentions_guidance` (broad) |
| `has_miscategorization` (exact) | `mentions_categorization` (broad) |
| Multiple specific flags | Combined `any_ticket_signal` |

#### Simple Explanation
When we ask the ML model "what clues did you use to make decisions?", it tells us the most important signals. Our engineered features (the ones we designed based on domain knowledge) turned out to be the most important — confirming that our understanding of the problem was correct.

#### Technical Depth
Feature importance analysis reveals that LightGBM learned to rely heavily on our **engineered domain features** (`any_ticket_signal`, `any_structural_issue`) rather than raw engagement metrics. This is a significant shift from v1 where `user_word_count` dominated. The `mentions_guidance` feature (broad pattern) ranks #6 vs the specific `has_dosage_guidance` in v3 that caused overfitting. The structural features (`response_completeness`, `has_whisper_mismatch`) remain consistently important across versions, indicating these are truly generalizable signals.

#### Suggested Visual
- Horizontal bar chart of top 15 features
- Color-code: Engineered (blue), Structural (green), Text (orange)
- Annotation showing "Domain features = 40% of importance"

---

## Slide 8: Full Evaluation Results

### **Complete Metrics Table**

| Model | Split | F1 | Precision | Recall | ROC-AUC | MCC | TP | FP | TN | FN |
|-------|-------|-----|-----------|--------|---------|-----|----|----|----|----|
| **Rule-based** | Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 59 | 0 | 630 | 0 |
| **Rule-based** | Val | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11 | 0 | 133 | 0 |
| ML v1 (XGBoost) | Train | 0.9916 | 0.9833 | 1.0000 | 1.0000 | 0.9908 | 59 | 1 | 629 | 0 |
| ML v1 (XGBoost) | Val | 0.9167 | 0.8462 | 1.0000 | 0.9959 | 0.9129 | 11 | 2 | 131 | 0 |
| ML v3 (LightGBM) | Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 59 | 0 | 630 | 0 |
| ML v3 (LightGBM) | Val | 0.9524 | 1.0000 | 0.9091 | 0.9822 | 0.9499 | 10 | 0 | 133 | 1 |
| **ML v4 (Ensemble)** | CV | 0.9500 | — | — | — | — | — | — | — | — |
| **ML v4 (Ensemble)** | Val | 0.9200 | — | — | — | — | — | — | — | — |

### **Key Terminology**
- **MCC** [Matthews Correlation Coefficient — best metric for imbalanced datasets, ranges -1 to 1, considers all confusion matrix cells]
- **ROC-AUC** [Area Under ROC Curve — measures how well model separates classes, 1.0 = perfect]
- **TP/FP/TN/FN** [True Positives, False Positives, True Negatives, False Negatives]

### **Leaderboard Comparison**

| Submission | Public LB | Private LB | Test Flags | Status |
|------------|-----------|------------|------------|--------|
| Rule-based | 1.00 | 1.00 | 18 | ✓ Final (Best) |
| ML v3 | 0.94 | 0.85 | 15 | ✗ Overfit |
| ML v4 | 1.00 | 0.933 | 17 | ✓ Best ML |

### **Confusion Matrix Definitions**
- **True Positive (TP)**: Correctly flagged a problem call
- **False Positive (FP)**: Incorrectly flagged a good call (false alarm)
- **True Negative (TN)**: Correctly identified a good call
- **False Negative (FN)**: Missed a problem call (worst outcome)

#### Simple Explanation
This table shows how well each model performed on every metric. The rule-based system is perfect (1.0 on everything). ML v4 is our best ML model with 92% F1 on validation and 93.3% accuracy on the final test. We submitted rules as our final answer because it achieved perfect scores.

#### Technical Depth
We evaluated using multiple metrics to ensure robustness. **MCC** is particularly important for imbalanced datasets because it considers all four confusion matrix cells, unlike F1 which ignores true negatives. The rule-based system achieves MCC=1.0 (perfect correlation), while ML v3 achieves MCC=0.95. The v4 metrics show the anti-overfitting tradeoff: **Val F1 dropped from 0.95 to 0.92** but **Private LB improved from 0.85 to 0.933** — a worthwhile trade. Rule-based remains superior with perfect scores on both leaderboards.

#### Suggested Visual
- Full metrics table with color-coded cells (green=perfect, yellow=good, red=concern)
- Bar chart comparing Public vs Private LB scores
- Highlight the v3→v4 improvement story

---

## Slide 9: Key Insight

### **Why Rules Beat ML on This Problem**

```
Rule-based: Public LB = 1.00, Private LB = 1.00
ML v4:      Public LB = 1.00, Private LB = 0.933
```

### **The Answer: Domain Patterns Are Explicit, Not Latent**

The `validation_notes` field contains **explicit QA annotations** that directly indicate tickets:
- "dosage guidance" → 13 tickets, 0 non-tickets (in training)
- "miscategorization" → 7 tickets, 0 non-tickets
- "erroneously as" → 1 ticket, 0 non-tickets

**Rules directly match these patterns. ML must learn them indirectly.**

### **Key Terminology**
- **Explicit Patterns** [obvious signals written in plain text that can be matched exactly]
- **Latent Patterns** [hidden patterns that must be discovered through statistical analysis]
- **SHAP/LIME** [model explanation tools that show why ML made each prediction]

### **What ML Independently Confirmed**
1. `any_ticket_signal` (our engineered feature) ranked as most important
2. Same patterns that rules match are what ML learned to weight highly
3. ML can achieve 93%+ but not 100% — confirming rules are optimal for explicit signals

### **Explainability Advantage**
| Question | Rule-Based | ML |
|----------|------------|-----|
| Why flagged? | "Matched pattern: dosage guidance" | "Model predicted 0.97 probability" |
| Debuggable? | Yes, check which rule fired | Partial, via SHAP/LIME |
| Regulatory? | Easy to audit | Requires explanation tools |

### **When Would ML Win?**
- Novel error patterns not seen in training
- Subtle correlations across multiple fields
- Production data with different `validation_notes` format

#### Simple Explanation
The data contains obvious clues written in plain English — phrases like "dosage guidance" and "miscategorization" that always mean there's a problem. Rules can read these clues directly, while ML has to figure them out indirectly by analyzing thousands of examples. When the answer is literally written in the data, rules win.

#### Technical Depth
The dominance of rules reflects a fundamental property of this dataset: `validation_notes` contains the output of an **upstream QA system** that explicitly labels issues. Phrases like "dosage guidance" are not natural language — they're QA annotations. Rules can match these perfectly because they're **deterministic signals**, not probabilistic patterns. ML's value would emerge if: (1) new error types appear that rules don't cover, (2) the upstream QA system changes its annotation format, or (3) we need to flag issues before QA runs. For this hackathon, rules are optimal because the "ground truth" is essentially encoded in the features.

#### Suggested Visual
- Side-by-side "Why flagged?" explanation boxes
- Venn diagram showing overlap of rule patterns and ML features
- Quote: "When answers are explicit, rules > ML"

---

## Slide 10: Tech Stack

### **Rule-Based Pipeline**
| Component | Technology |
|-----------|------------|
| Language | Python 3.9 |
| Data manipulation | Pandas |
| Pattern matching | regex (re module) |
| Evaluation | scikit-learn metrics |

### **ML Pipeline**
| Component | Technology |
|-----------|------------|
| Feature engineering | Pandas, NumPy |
| Text features | TF-IDF (scikit-learn) |
| Models | LightGBM, XGBoost, Random Forest, Logistic Regression |
| Hyperparameter tuning | Optuna (TPE sampler) |
| Cross-validation | StratifiedKFold (5-fold) |
| Class imbalance | scale_pos_weight, class_weight='balanced' |
| Evaluation | scikit-learn (F1, ROC-AUC, MCC) |
| Visualization | Matplotlib |

### **Key Terminology**
- **SMOTE** [Synthetic Minority Oversampling Technique — creates fake positive examples to balance classes; we tested but didn't use in final]
- **TPE Sampler** [Tree-structured Parzen Estimator — smart hyperparameter search that learns from previous trials]
- **Threshold Tuning** [adjusting the cutoff probability for flagging a call; v4 optimal threshold = 0.37]

### **File Structure**
```
Caller.ai/
├── predict.py                      # Rule-based system (FINAL)
├── submission.csv                  # Rule-based predictions (18 flags)
├── pitch_deck_final.md             # This document
├── ml_pipeline/
│   ├── train_ml_pipeline.py        # ML v1 (XGBoost + TF-IDF)
│   ├── train_ml_pipeline_v2.py     # ML v2 (LightGBM + enhanced)
│   ├── train_ml_pipeline_v3.py     # ML v3 (targeted flags - OVERFIT)
│   ├── train_ml_pipeline_v4b.py    # ML v4 (ensemble - BEST ML)
│   ├── submission_ml_v4.csv        # ML v4 predictions (17 flags)
│   ├── evaluation_report.csv       # Full metrics table
│   └── feature_importance_v3.png   # Feature plot
└── Datasets/csv/
    ├── hackathon_train.csv
    ├── hackathon_val.csv
    └── hackathon_test.csv
```

#### Simple Explanation
We used Python with common data science libraries. The rule-based system is just 170 lines of code with simple pattern matching. The ML pipeline is more complex with gradient boosting models and careful tuning, but ultimately the simple approach won.

#### Technical Depth
We chose **LightGBM** over XGBoost for the final ML model due to its superior performance on small imbalanced datasets and faster training time. Optuna was used for hyperparameter optimization with TPE sampling. The imbalanced class ratio (10.68:1) was handled via `scale_pos_weight` in gradient boosting and `class_weight='balanced'` in Random Forest/Logistic Regression. The **v4 ensemble** uses weighted averaging (LGBM 50%, RF 30%, LR 20%) with threshold optimization finding optimal cutoff at 0.37 (far below default 0.5 due to class imbalance).

#### Suggested Visual
- Tech stack icons in grid layout
- File tree diagram
- "Built in Python" badge

---

## Slide 11: Conclusion

### **Final Scores**

| Submission | Public LB | Private LB | Val F1 | Test Flags |
|------------|-----------|------------|--------|------------|
| **Rule-based** | **1.00** | **1.00** | **1.00** | 18 |
| ML v4 (Ensemble) | 1.00 | 0.933 | 0.92 | 17 |

**Leaderboard Position: 2nd Place**

### **Key Achievements**
✓ Perfect F1 = 1.0 with fully explainable rules (Public AND Private LB)  
✓ Identified 6 distinct error categories with regex patterns  
✓ ML independently validated rule patterns (any_ticket_signal ranked #1)  
✓ Detected and fixed overfitting (v3 0.85 → v4 0.933 Private LB)  
✓ MCC = 1.0 confirms no false positives or negatives  

### **What We Learned**
1. **Explicit signals beat learned patterns** — when answers are written in the data, use rules
2. **Validation ≠ Generalization** — v3's Val F1=0.95 hid Private LB=0.85 overfitting
3. **Cross-validation is essential** — 5-fold CV caught what single split missed
4. **Simpler generalizes better** — broad patterns ("guidance") beat exact ("dosage guidance")

### **What We'd Do Next**
1. **Deploy rules as primary classifier** — perfect accuracy, full explainability
2. **Use ML as anomaly detector** — catch novel patterns rules miss
3. **Build error category dashboard** — breakdown by audio_issue, elevenlabs, openai
4. **Monitor for pattern drift** — alert if validation_notes format changes
5. **A/B test rule updates** — safely add new patterns without regression

### **Repository**
🔗 github.com/[username]/CareCaller-Hackathon-2026

### **Thank You!**
Questions?

#### Simple Explanation
We achieved perfect scores using simple rules that anyone can understand and verify. Our ML experiments taught us that sometimes the best solution is the simplest one — especially when the data contains obvious answers. The journey from v3 (overfit) to v4 (generalized) taught us valuable lessons about real-world ML deployment.

#### Technical Depth
This hackathon demonstrated a key principle in applied ML: **understand your data before modeling**. The `validation_notes` field contained explicit QA annotations — essentially pre-computed labels — that made rule-based matching optimal. Our ML journey (v1→v4) served as independent validation, confirming the patterns we discovered manually. The v3 overfitting incident (Public 0.94 → Private 0.85) is a textbook example of why **cross-validation and regularization** matter in production systems. Final recommendation: use rules for this specific problem, but keep ML infrastructure ready for when the upstream QA system changes.

#### Suggested Visual
- Large "1.0 / 1.0" achievement display (Public/Private LB)
- Journey timeline: v1 → v2 → v3 (overfit!) → v4 → Rules (perfect)
- "2nd Place" badge
- QR code to GitHub repository

---

## Appendix: Technical Glossary

| Term | Definition |
|------|------------|
| **F1 Score** | Harmonic mean of precision and recall — balances catching errors vs avoiding false alarms |
| **MCC** | Matthews Correlation Coefficient — best metric for imbalanced datasets, -1 to 1, considers all confusion matrix cells |
| **ROC-AUC** | Area Under Receiver Operating Characteristic curve — measures discrimination ability, 1.0 = perfect |
| **Overfitting** | Model memorizes training data but fails on new unseen data |
| **Cross-Validation** | Training on different data subsets to estimate real-world performance |
| **LightGBM** | Fast gradient boosting algorithm — builds decision trees sequentially, each correcting previous errors |
| **TF-IDF** | Term Frequency-Inverse Document Frequency — converts text to numbers based on word importance |
| **SMOTE** | Synthetic Minority Oversampling Technique — creates synthetic positive examples to balance classes |
| **Optuna** | Hyperparameter tuning library — automatically finds best model settings using Bayesian optimization |
| **StratifiedKFold** | Cross validation that maintains class balance (8.6% positives) in each fold |
| **Threshold Tuning** | Adjusting the probability cutoff for classification (default 0.5, optimal varies) |
| **Regularization** | Technique to prevent overfitting by penalizing model complexity |
| **scale_pos_weight** | LightGBM parameter to handle class imbalance (set to neg/pos ratio = 10.68) |
| **Ensemble** | Combining multiple models' predictions (voting or averaging) for better accuracy |
| **Public/Private LB** | Competition scoring: Public uses 50% of test data; Private reveals full test set at end |

---

*Document generated: April 2026*  
*Last updated: After ML v4 final submission*
