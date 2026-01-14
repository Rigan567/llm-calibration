# src_combined/inference_openai.py

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

# =====================================================
# Setup
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================
# Files & model  🔧 CHANGED
# =====================================================
INPUT_FILE = "dataset/combined_qa_dataset_800.jsonl"
OUTPUT_CSV = "outputs/baseline_gpt_oss_120b.csv"   # 🔧 CHANGED
MODEL_NAME = "openai/gpt-oss-120b"                 # 🔧 CHANGED
PROMPT_VERSION = "baseline"

# =====================================================
# Experiment control
# =====================================================
TARGET_SOURCES = {
    "Astro-QA_Judgement",
    "HotpotQA",
    "GlobalMedQA_EN",
    "TemporalQA"
}

SAMPLES_PER_SOURCE = 2

# =====================================================
# Output parsing (unchanged)
# =====================================================
def parse_model_output(text):
    lines = [line for line in text.splitlines() if line.strip() != '']
    answer = lines[-2]
    try:
        confidence = float(lines[-1].strip())
    except ValueError:
        confidence = None
    return answer, confidence, text

# =====================================================
# Main inference loop
# =====================================================
def main():
    df = pd.read_json(INPUT_FILE, lines=True)

    if "source" not in df.columns:
        raise ValueError("'source' column not found.")

    df = df[df["source"].isin(TARGET_SOURCES)]

    df = (
        df.groupby("source", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_SOURCE)))
          .reset_index(drop=True)
    )

    print("Selected samples per category:")
    print(df["source"].value_counts())

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        gold = row["answer"]
        source = row["source"]

        q_type = {
            "Open ended": "open",
            "True or False": "true_false",
            "Multiple-choice": "choice",
            "temporal": "temporal",
        }.get(row["type"])

        if q_type is None:
            continue

        prompt_template = open(
            BASE_DIR / f"prompts_openai/{PROMPT_VERSION}/{PROMPT_VERSION}_{q_type}.txt"
        ).read()

        prompt = prompt_template.format(question=question)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )

        text = response.choices[0].message.content.strip()
        pred, conf, raw = parse_model_output(text)

        rows.append({
            "question": question,
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "source": source,
            "raw_response": raw
        })

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved results -> {OUTPUT_CSV}")

# =====================================================
if __name__ == "__main__":
    main()
