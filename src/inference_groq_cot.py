# src/inference_groq_cot.py

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import re
from tqdm import tqdm

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/processed/hotpot_clean.jsonl"
OUTPUT_CSV = "outputs/baseline_groq_cot.csv"
PROMPT_TEMPLATE = open("prompts/cot.txt").read()

# MODEL_NAME = "llama-3.1-8b-instant"  # working model
MODEL_NAME = "llama-3.3-70b-versatile"  # working model

def parse_confidence(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last = lines[-1]
    m = re.match(r"^0(\.\d+)?$|^1(\.0+)?$", last)
    return float(last) if m else None

def parse_answer(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines[-2]
    return lines[-1]

def main():
    df = pd.read_json(INPUT_FILE, lines=True).head(20)
    os.makedirs("outputs", exist_ok=True)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = PROMPT_TEMPLATE.format(
            context=row["context"],
            question=row["question"]
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content
        pred = parse_answer(text)
        confidence = parse_confidence(text)

        rows.append({
            "question": row["question"],
            "gold": row["answer"],
            "pred": pred,
            "confidence": confidence,
            "raw": text
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print("Saved ->", OUTPUT_CSV)

if __name__ == "__main__":
    main()
