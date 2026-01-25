import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch

# --------------------------------------------------
# Path setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.answer_matching import (
    exact_matches,
    f1_token_levels,
    bert_scores
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "src_gemma_27b" / "outputs"
OUTPUTS_DIR = PROJECT_ROOT / "evaluation_results" / "src_gemma_27b"

MODEL_TAG = "gemma-3-27b-it"

"""
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

# --------------------------------------------------
# Safe BERTScore
# --------------------------------------------------
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
        return None  # never crash

"""


# --------------------------------------------------
# Evaluate a single CSV with resume
# --------------------------------------------------
def evaluate_csv(csv_path: Path, output_path: Path, batch_size: int = 100):
    # 1. Read entire CSV
    df = pd.read_csv(csv_path)
    before = len(df)

    # 2. Drop rows with empty / missing required fields
    required_cols = ["gold", "pred", "confidence", "source"]

    df = (
        df[required_cols]
        .dropna()
        .loc[
            lambda x: (x["gold"].astype(str).str.strip() != "")
                      & (x["pred"].astype(str).str.strip() != "")
                      & (x["source"].astype(str).str.strip() != "")
        ]
        .reset_index(drop=True)
    )
    after = len(df)
    print(f"🧹 Dropped {before - after} invalid rows")

    if df.empty:
        print(f"⚠️ No valid rows left after filtering → {csv_path.name}")
        return

    all_em = []
    all_f1 = []
    all_bert = []

    num_rows = len(df)

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)

        batch = df.iloc[start:end]
        preds = batch["pred"].astype(str).tolist()
        golds = batch["gold"].astype(str).tolist()

        em = exact_matches(golds, preds)
        f1_scores = f1_token_levels(golds, preds)
        with torch.no_grad():
            bert_scs = bert_scores(golds, preds)

        all_em.extend(em)
        all_f1.extend(f1_scores)
        all_bert.extend(bert_scs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Build output DataFrame
    out_df = pd.DataFrame({
        "confidence": df["confidence"].values,
        "source": df["source"].values,
        "em": all_em,
        "f1": all_f1,
        "bert": all_bert,
    })

    # 4. Write results
    out_df.to_csv(output_path, index=False)

    print(f"✅ Saved batch evaluation → {output_path}")


# --------------------------------------------------
# Main (fully resumable)
# --------------------------------------------------
def main():
    csv_files = sorted(
        f for f in INPUTS_DIR.glob("*.csv")
        if MODEL_TAG in f.name #and f.name.endswith("_results.csv")
    )

    print(f"🔍 Found {len(csv_files)} result files to evaluate")

    for csv_file in csv_files:
        prompt = csv_file.stem.replace(f"_{MODEL_TAG}_results", "")
        output_path = OUTPUTS_DIR / f"{prompt}_{MODEL_TAG}_metrics.csv"

        if output_path.exists():
            print(f"⏭️ Skipping {prompt} (already evaluated)")
            continue

        print(f"📊 Evaluating {csv_file.name}")
        evaluate_csv(csv_file, output_path)
        print(f"✅ Saved metrics for {prompt}")

    print("\n🎉 Batch evaluation complete")


if __name__ == "__main__":
    main()
