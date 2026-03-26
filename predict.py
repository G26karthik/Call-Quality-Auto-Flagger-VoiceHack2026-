"""
CareCaller Hackathon 2026 — Problem 1: Call Quality Auto-Flagger
Rule-based predictor for has_ticket (calls needing human review).
"""

import json
import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_DIR = "Datasets/csv"

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Validation-notes keyword patterns → (regex, category)
VALIDATION_PATTERNS = [
    # STT / audio issues → audio_issue
    (r"weight differs between sources", "audio_issue"),
    (r"erroneously as", "audio_issue"),
    (r"stt error", "audio_issue"),
    (r"weight discrepancy", "audio_issue"),
    # Medical advice / guardrail violations → elevenlabs
    (r"dosage guidance", "elevenlabs"),
    (r"guardrail", "elevenlabs"),
    (r"medical advice", "elevenlabs"),
    (r"vitamin supplementation", "elevenlabs"),
    # Outcome mismatch → openai
    (r"miscategor", "openai"),
    (r"does not match", "openai"),
    (r"corrected by validation", "openai"),
    (r"classified as wrong_number", "openai"),
    # Data contradiction → openai
    (r"inconsistency noted", "openai"),
    (r"inconsistency detected", "openai"),
    (r"allergy.*but recorded", "openai"),
    (r"noted discrepancy", "openai"),
    # Agent skipped questions → openai
    (r"fabricated responses as questions were not asked", "openai"),
    # Wrong-number misclassification → openai
    (r"confirmed.*identity.*wrong_number", "openai"),
    (r"confirmed identity but stated.*do not want", "openai"),
    # Opted-out miscategorization → openai
    (r"expressed interest but", "openai"),
    # Response contradicts transcript → openai
    (r"but recorded response indicates", "openai"),
    # Unsolicited medical recommendation → elevenlabs
    (r"recommendations provided for", "elevenlabs"),
]


def _has_side_effect_contradiction(transcript: str, resp_json: str) -> bool:
    """Check if responses_json records side effects the patient denied in the transcript."""
    if not resp_json:
        return False
    try:
        responses = json.loads(resp_json)
    except (json.JSONDecodeError, TypeError):
        return False

    # Find the side-effects answer in responses_json
    for r in responses:
        q = str(r.get("question", "")).lower()
        if "side effect" not in q:
            continue
        answer = str(r.get("answer", "")).lower().strip()
        if "yes" not in answer and "nausea" not in answer:
            return False  # recorded as no → no contradiction
        # Recorded says yes/nausea — check what user actually said
        m = re.search(
            r'side effect[^[]*\[USER\]:\s*([^[]+)',
            transcript, re.IGNORECASE
        )
        if m:
            user_said = m.group(1).strip().lower()
            if re.search(r'\bno\b|not really|none|haven\'t', user_said) and \
               not re.search(r'\byes\b|yeah|nausea', user_said):
                return True
    return False


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all flagging rules. Returns df with predicted_ticket and predicted_category."""

    n = len(df)
    flagged = [False] * n
    categories = [""] * n

    notes = df["validation_notes"].fillna("").str.lower().tolist()
    whisper_mm = df["whisper_mismatch_count"].fillna(0).astype(int).tolist()
    outcome = df["outcome"].fillna("").str.lower().tolist()
    resp_comp = df["response_completeness"].fillna(1.0).astype(float).tolist()
    avg_user_words = df["avg_user_turn_words"].fillna(0).astype(float).tolist()
    user_word_count = df["user_word_count"].fillna(0).astype(int).tolist()
    transcripts = df["transcript_text"].fillna("").tolist()
    responses = df["responses_json"].fillna("").tolist()

    for i in range(n):
        note = notes[i]
        cat = ""

        # --- Validation-notes patterns ---
        for pattern, pattern_cat in VALIDATION_PATTERNS:
            if re.search(pattern, note):
                flagged[i] = True
                # Keep first category matched (priority order in list)
                if not cat:
                    cat = pattern_cat

        # --- Structural rules ---
        # whisper_mismatch_count > 0 → always ticket (audio_issue)
        if whisper_mm[i] > 0:
            flagged[i] = True
            if not cat:
                cat = "audio_issue"

        # outcome == completed AND 0 < response_completeness < 1.0 → always ticket
        if outcome[i] == "completed" and 0 < resp_comp[i] < 1.0:
            flagged[i] = True
            if not cat:
                cat = "openai"

        # outcome == incomplete AND avg_user_turn_words > 8
        #   AND response_completeness < 0.65 AND user_word_count > 80 → likely ticket
        if (
            outcome[i] == "incomplete"
            and avg_user_words[i] > 8
            and resp_comp[i] < 0.65
            and user_word_count[i] > 80
        ):
            flagged[i] = True
            if not cat:
                cat = "openai"

        # --- Transcript-level rule: side-effect contradiction ---
        # Response records "yes, mild nausea" but patient said no/not really
        if not flagged[i] or True:  # always check to assign category
            if _has_side_effect_contradiction(transcripts[i], responses[i]):
                flagged[i] = True
                if not cat:
                    cat = "openai"

        categories[i] = cat

    df = df.copy()
    df["predicted_ticket"] = flagged
    df["predicted_category"] = categories
    return df


def evaluate(df: pd.DataFrame, split_name: str) -> None:
    """Print metrics for a labeled split."""
    y_true = df["has_ticket"].astype(bool)
    y_pred = df["predicted_ticket"].astype(bool)

    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation")
    print(f"{'='*50}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*50}")

    # Show any false negatives for debugging
    if fn > 0:
        missed = df[y_true & ~y_pred][["call_id", "outcome", "whisper_mismatch_count",
                                        "response_completeness", "validation_notes"]]
        print(f"\n  Missed cases (FN={fn}):")
        for _, row in missed.iterrows():
            print(f"    call_id={row['call_id']}")
            print(f"      outcome={row['outcome']}, whisper_mm={row['whisper_mismatch_count']}, "
                  f"resp_comp={row['response_completeness']}")
            vnote = str(row['validation_notes'])[:120]
            print(f"      notes={vnote}...")


def main():
    # Load datasets
    train = pd.read_csv(f"{DATA_DIR}/hackathon_train.csv")
    val = pd.read_csv(f"{DATA_DIR}/hackathon_val.csv")
    test = pd.read_csv(f"{DATA_DIR}/hackathon_test.csv")

    # Apply rules
    train = apply_rules(train)
    val = apply_rules(val)
    test = apply_rules(test)

    # Evaluate on labeled splits
    evaluate(train, "TRAIN")
    evaluate(val, "VALIDATION")

    # Generate submission (Kaggle format: call_id, has_ticket as 0/1)
    submission = test[["call_id", "predicted_ticket"]].copy()
    submission = submission.rename(columns={"predicted_ticket": "has_ticket"})
    submission["has_ticket"] = submission["has_ticket"].astype(int)
    submission.to_csv("submission.csv", index=False)

    print(f"\nSubmission saved: submission.csv  ({len(submission)} rows)")
    print(f"  Predicted tickets: {submission['has_ticket'].sum()}")
    print(f"  Category breakdown:")
    cats = test[test["predicted_ticket"]]["predicted_category"].value_counts()
    for cat, count in cats.items():
        print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()
