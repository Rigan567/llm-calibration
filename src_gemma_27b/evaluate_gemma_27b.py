import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from utils.answer_matching import (
    exact_matches,
    f1_token_levels,
    bert_scores
)

from utils.calibration import (
    compute_brier,
    compute_ece
)

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EVAL_DIR = BASE_DIR / "eval_results"

EVAL_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = EVAL_DIR / "gemma_27b_summary.csv"

MODEL_TAG = "gemma-3-27b-it"

# =====================================================
# Per-file evaluation
# =====================================================
def evaluate_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    em = []
    f1 = []
    bert_f1 = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_path.name):
        pred = str(row["pred"])
        gold = str(row["gold"])

        # em.append(exact_matches(pred, gold))
        # f1.append(f1_token_levels(pred, gold))
        # bert_f1.append(bert_scores([pred], gold))

        em.append(exact_matches([pred], [gold])[0])
        f1.append(f1_token_levels([pred], [gold])[0])
        bert_f1.append(bert_scores([pred], [gold])[0])


    accuracy = sum(em) / len(em)
    avg_token_f1 = sum(f1) / len(f1)
    avg_bert_f1 = sum(bert_f1) / len(bert_f1)

    confidences = df["confidence"].fillna(0.0).tolist()
    correctness = em

    brier = compute_brier(confidences, correctness)
    ece = compute_ece(confidences, correctness)

    prompt_name = csv_path.stem.replace(f"_{MODEL_TAG}", "")

    return {
        "prompt": prompt_name,
        "accuracy": accuracy,
        "avg_token_f1": avg_token_f1,
        "avg_bertscore_f1": avg_bert_f1,
        "brier_score": brier,
        "ece": ece,
        "num_samples": len(df),
    }

# =====================================================
# Main
# =====================================================
def main():
    rows = []

    csv_files = sorted(
        f for f in OUTPUTS_DIR.glob("*.csv")
        if MODEL_TAG in f.name
    )

    if not csv_files:
        raise RuntimeError("No Gemma-27B output CSVs found.")

    for csv_file in csv_files:
        print(f"Evaluating {csv_file.name}")
        rows.append(evaluate_csv(csv_file))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved evaluation summary -> {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
