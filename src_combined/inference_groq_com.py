# src_combined/inference_groq_com.py

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

#setup
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/processed/combined_clean.jsonl"
OUTPUT_CSV = "outputs/baseline_groq.csv"
MODEL_NAME = "llama-3.1-8b-instant"
PROMPT_TEMPLATE = open("prompts/baseline.txt").read()


# Experiment control
# Choosing 5 categories (sources)
TARGET_SOURCES = {
    "Astro-QA_Judgement",   # True / False
    "HotpotQA",             # Multi-hop factoid
    "GlobalMedQA_EN",       # Medical MCQ
    # "TORQUE",               # Event / temporal
    "TemporalQA"            # Temporal reasoning (if present)
}

SAMPLES_PER_SOURCE = 2




def parse_model_output(text):
    lines = [line for line in text.splitlines() if line.strip() != '']
    answer = lines[-2]
    try:
        confidence = float(lines[-1].strip())
    except ValueError:
        confidence = None
    return answer, confidence, text


# Main inference loop
def main():
    # Load processed dataset
    df = pd.read_json(INPUT_FILE, lines=True)

    # Ensure source exists
    if "source" not in df.columns:
        raise ValueError(
            "'source' column not found. "
        )

    # Keep only chosen categories
    df = df[df["source"].isin(TARGET_SOURCES)]

    # Sample equally from each category
    df = (
        df.groupby("source", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_SOURCE)))
          .reset_index(drop=True)
    )

    print("Selected samples per category:")
    print(df["source"].value_counts())

    rows = []

    # Run inference
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        gold = row["answer"]
        source = row["source"]

        prompt = PROMPT_TEMPLATE.format(question=question)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
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

    # Save results
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved results -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
