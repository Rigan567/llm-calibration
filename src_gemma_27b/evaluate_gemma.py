import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.answer_matching import (
    exact_matches,
    f1_token_levels,
    bert_scores
)

INPUT_CSV = "outputs/baseline_gemma.csv"
NUM_BINS = 10

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

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

def main():
    df = pd.read_csv(INPUT_CSV)
    # df["confidence"] = df["confidence"].fillna(0.5)
    # df = df.dropna()

    df = df.dropna(subset=["pred", "gold", "confidence"])
    df["confidence"] = df["confidence"].astype(float)

    preds= df["pred"].astype(str).tolist()
    golds= df["gold"].astype(str).tolist()

    # Answer quality
    df["exact_match"] = exact_matches(golds, preds)
    df["token_f1"] = f1_token_levels(golds, preds)
    df["bertscore"] = bert_scores(golds, preds)

    probs = df["confidence"].values
    correct = df["bertscore"].values

    print("=== Evaluation ===")
    print(f"Exact-Match Accuracy   : {df['exact_match'].mean():.3f}")
    print(f"Avg Token F1           : {df['token_f1'].mean():.3f}")
    print(f"Avg BERTScore F1       : {df['bertscore'].mean():.3f}")
    print("--- Calibration ---")
    
    print(f"Brier Score            : {brier_score(correct, probs):.3f}")
    print(f"ECE (10 bins)          : {compute_ece(probs, correct):.3f}")

    df.to_csv("outputs/eval_results_gemma.csv", index=False)
    print("Saved detailed results -> outputs/eval_results_gemma.csv")

if __name__ == "__main__":
    main()
