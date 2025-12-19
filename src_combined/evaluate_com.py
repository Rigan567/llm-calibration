# src/evaluate_com.py

import pandas as pd
import numpy as np
import re
import os

# Import your new matching functions
from answer_matching import (
    exact_match,
    f1_token_level,
    bert_score
)

INPUT_CSV = "outputs/baseline_groq.csv"
NUM_BINS = 10


# ------------------------------------------------
# Normalization for fallback semantic checks
# ------------------------------------------------
def normalize(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


# ------------------------------------------------
# Calibration: Brier Score
# ------------------------------------------------
def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)


# ------------------------------------------------
# Calibration: Expected Calibration Error
# ------------------------------------------------
def compute_ece(probs, correct, num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue

        avg_conf = probs[mask].mean()
        avg_acc = correct[mask].mean()

        ece += abs(avg_conf - avg_acc) * (mask.sum() / len(probs))

    return ece


# ------------------------------------------------
# Main Evaluation Pipeline
# ------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    # Fix missing confidence
    df["confidence"] = df["confidence"].fillna(0.5)

    preds = df["pred"].astype(str).tolist()
    golds = df["gold"].astype(str).tolist()

    # ---------- TEXT MATCHING METRICS ----------
    df["exact_match"] = [
        exact_match(p, g) for p, g in zip(preds, golds)
    ]

    df["token_f1"] = [
        f1_token_level(p, g) for p, g in zip(preds, golds)
    ]

    df["bertscore"] = [
        bert_score(p, g) for p, g in zip(preds, golds)
    ]

    # Accuracy = mean of exact match
    acc = df["exact_match"].mean()

    # ---------- CALIBRATION METRICS ----------
    probs = df["confidence"].values
    correct_binary = df["exact_match"].values  # For calibration, binary needed

    brier = brier_score(correct_binary, probs)
    ece = compute_ece(probs, correct_binary, NUM_BINS)

    # ---------- PRINT RESULTS ----------
    print("=== Evaluation ===")
    print(f"Exact-Match Accuracy   : {acc:.3f}")
    print(f"Avg Token F1           : {df['token_f1'].mean():.3f}")
    print(f"Avg BERTScore F1       : {df['bertscore'].mean():.3f}")
    print("--- Calibration ---")
    print(f"Brier Score            : {brier:.3f}")
    print(f"ECE (10 bins)          : {ece:.3f}")

    # Save detailed result sheet
    df.to_csv("outputs/eval_results_detailed.csv", index=False)
    print("Saved detailed results -> outputs/eval_results_detailed.csv")


if __name__ == "__main__":
    main()
