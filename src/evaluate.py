# src/evaluate.py

import pandas as pd
import numpy as np
import re
import os

# INPUT_CSV = "outputs/baseline_groq_cot.csv"   # change for COT or Self-Consistency
INPUT_CSV = max(
    ["outputs/baseline_groq.csv",
     "outputs/baseline_groq_cot.csv",
     "outputs/self_consistency_groq.csv"],
    key=lambda x: os.path.getmtime(x)
)

NUM_BINS = 10

# ------------------------------
# Normalize text for comparison
# ------------------------------
def normalize(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

# ------------------------------
# Semantic matching for accuracy
# ------------------------------
def semantic_match(pred, gold):
    """
    Flexible matching:
    - handles yes/no questions
    - checks substring overlaps
    - checks numeric answers
    """

    p = normalize(pred)
    g = normalize(gold)

    if g == "" or p == "":
        return False

    # 1) Yes/No questions
    if g in ["yes", "no"]:
        return g in p

    # 2) Numeric answers (e.g., "3677")
    if g.isdigit():
        return g in p

    # 3) Gold answer is fully contained in prediction
    if g in p:
        return True

    # 4) Prediction contained inside gold (rare)
    if p in g:
        return True

    return False

# ------------------------------
# Brier Score
# ------------------------------
def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

# ------------------------------
# Expected Calibration Error
# ------------------------------
def compute_ece(probs, correct, num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        left, right = bins[i], bins[i+1]
        mask = (probs >= left) & (probs < right)

        if mask.sum() == 0:
            continue
        
        avg_conf = probs[mask].mean()
        avg_acc = correct[mask].mean()

        ece += abs(avg_conf - avg_acc) * (mask.sum() / len(probs))
    
    return ece

# ------------------------------
# Main evaluation
# ------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    # Fix missing confidence
    df["confidence"] = df["confidence"].fillna(0.5)

    # Compute semantic correctness
    df["correct"] = df.apply(
        lambda row: semantic_match(row["pred"], row["gold"]), axis=1
    ).astype(int)

    probs = df["confidence"].values
    correct = df["correct"].values

    acc = correct.mean()
    brier = brier_score(correct, probs)
    ece = compute_ece(probs, correct, NUM_BINS)

    print("=== Evaluation ===")
    print(f"Accuracy       : {acc:.3f}")
    print(f"Brier Score    : {brier:.3f}")
    print(f"ECE (10 bins)  : {ece:.3f}")

    df.to_csv("outputs/eval_results.csv", index=False)
    print("Saved detailed results -> outputs/eval_results.csv")

if __name__ == "__main__":
    main()
