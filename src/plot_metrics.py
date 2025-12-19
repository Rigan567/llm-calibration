# src/plot_metrics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_CSV = "outputs/eval_results_detailed.csv"
OUT_DIR = "outputs/plots"

os.makedirs(OUT_DIR, exist_ok=True)


def plot_accuracy_bars(df):
    """Bar chart comparing accuracy metrics."""
    metrics = ["exact_match", "token_f1", "bertscore"]
    values = [df["exact_match"].mean(),
              df["token_f1"].mean(),
              df["bertscore"].mean()]

    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values, color=["#4E79A7", "#F28E2B", "#59A14F"])
    plt.ylim(0, 1)
    plt.title("Accuracy Metrics Comparison")
    plt.ylabel("Score")
    plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=300)
    plt.close()


def plot_confidence_hist(df):
    """Histogram of model confidence."""
    plt.figure(figsize=(7, 5))
    plt.hist(df["confidence"], bins=20, color="#4E79A7", alpha=0.7)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUT_DIR}/confidence_histogram.png", dpi=300)
    plt.close()


def plot_reliability_curve(df):
    """Calibration reliability curve (ECE visualization)."""
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)

    df["bin"] = np.digitize(df["confidence"], bins) - 1
    bin_acc = []
    bin_conf = []

    for b in range(num_bins):
        subset = df[df["bin"] == b]
        if len(subset) == 0:
            continue
        bin_conf.append(subset["confidence"].mean())
        bin_acc.append(subset["exact_match"].mean())

    plt.figure(figsize=(7, 5))
    plt.plot(bin_conf, bin_acc, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.title("Reliability Curve")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/reliability_curve.png", dpi=300)
    plt.close()


def plot_conf_vs_bert(df):
    """Scatter plot of confidence vs BERTScore F1."""
    plt.figure(figsize=(7, 5))
    plt.scatter(df["confidence"], df["bertscore"], alpha=0.6, color="#F28E2B")
    plt.xlabel("Confidence")
    plt.ylabel("BERTScore F1")
    plt.title("Confidence vs Semantic Quality")
    plt.savefig(f"{OUT_DIR}/conf_vs_bert.png", dpi=300)
    plt.close()


def plot_bert_box(df):
    """Boxplot of BERTScore distribution."""
    plt.figure(figsize=(5, 6))
    plt.boxplot(df["bertscore"], vert=True)
    plt.ylabel("BERTScore F1")
    plt.title("BERTScore Distribution")
    plt.savefig(f"{OUT_DIR}/bert_score_boxplot.png", dpi=300)
    plt.close()


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Generating plots...")

    plot_accuracy_bars(df)
    plot_confidence_hist(df)
    plot_reliability_curve(df)
    plot_conf_vs_bert(df)
    plot_bert_box(df)

    print(f"All plots saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
