import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch

# -----------------------------------------------------
# Project path setup
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.answer_matching import (
    exact_matches,
    f1_token_levels,
)

from bert_score import score as bert_score_fn

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

EVAL_DIR = BASE_DIR / "eval_results"
PARTIAL_DIR = EVAL_DIR / "gemma_partial"

EVAL_DIR.mkdir(exist_ok=True)
PARTIAL_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = EVAL_DIR / "gemma_27b_summary.csv"
MODEL_TAG = "gemma-3-27b-it"


def compute_ece(confidences, correctness, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    ece = 0.0
    N = len(confidences)

    for low, high in bins:
        idxs = [
            i for i, p in enumerate(confidences)
            if (p >= low and p < high) or (high == 1.0 and p == 1.0)
        ]
        if not idxs:
            continue

        acc = sum(correctness[i] for i in idxs) / len(idxs)
        conf = sum(confidences[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / N) * abs(acc - conf)

    return ece

# -----------------------------------------------------
# Safe BERTScore (never crash)
# -----------------------------------------------------
def safe_bertscore(pred, gold):
    try:
        P, R, F1 = bert_score_fn(
            [pred],
            [gold],
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            verbose=False,
        )
        return float(F1[0])
    except Exception:
        return None


# -----------------------------------------------------
# Per-CSV evaluation with resume
# -----------------------------------------------------
def evaluate_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    partial_path = PARTIAL_DIR / f"{csv_path.stem}.partial.csv"

    if partial_path.exists():
        partial_df = pd.read_csv(partial_path)
        start_idx = len(partial_df)
        print(f"🔄 Resuming {csv_path.name} from row {start_idx}")
    else:
        partial_df = pd.DataFrame(
            columns=["em", "f1", "bert_f1"]
        )
        start_idx = 0

    for i in tqdm(
        range(start_idx, len(df)),
        desc=csv_path.name,
        initial=start_idx,
        total=len(df),
    ):
        row = df.iloc[i]
        pred = str(row["pred"])
        gold = str(row["gold"])

        result = {
            "em": exact_matches(pred, gold),
            "f1": f1_token_levels(pred, gold),
            "bert_f1": safe_bertscore(pred, gold),
        }

        partial_df.loc[i] = result
        partial_df.to_csv(partial_path, index=False)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Aggregate metrics
    em = partial_df["em"].tolist()
    f1 = partial_df["f1"].tolist()
    bert = partial_df["bert_f1"].dropna().tolist()

    accuracy = sum(em) / len(em)
    avg_token_f1 = sum(f1) / len(f1)
    avg_bert_f1 = sum(bert) / len(bert) if bert else None

    confidences = df["confidence"].fillna(0.0).tolist()
    brier = sum((p - y) ** 2 for p, y in zip(confidences, em)) / len(em)

    prompt_name = csv_path.stem.replace(f"_{MODEL_TAG}", "")

    return {
        "prompt": prompt_name,
        "accuracy": accuracy,
        "avg_token_f1": avg_token_f1,
        "avg_bertscore_f1": avg_bert_f1,
        "brier_score": brier,
        "num_samples": len(df),
    }


# -----------------------------------------------------
# Main (CSV-level resume)
# -----------------------------------------------------
def main():
    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        completed = set(summary_df["prompt"].tolist())
        print(f"🔄 Found {len(completed)} completed prompts")
    else:
        summary_df = pd.DataFrame()
        completed = set()

    csv_files = sorted(
        f for f in OUTPUTS_DIR.glob("*.csv")
        if MODEL_TAG in f.name
    )

    if not csv_files:
        raise RuntimeError("No Gemma-4B output CSVs found.")

    for csv_file in csv_files:
        prompt_name = csv_file.stem.replace(f"_{MODEL_TAG}", "")

        if prompt_name in completed:
            print(f"⏭ Skipping {prompt_name}")
            continue

        print(f"📊 Evaluating {csv_file.name}")
        row = evaluate_csv(csv_file)

        summary_df = pd.concat(
            [summary_df, pd.DataFrame([row])],
            ignore_index=True
        )
        summary_df.to_csv(SUMMARY_CSV, index=False)

        torch.cuda.empty_cache()
        print(f"✅ Saved results for {prompt_name}")

    print(f"\n🎉 Evaluation complete → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
