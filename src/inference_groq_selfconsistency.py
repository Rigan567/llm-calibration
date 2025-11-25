# src/inference_groq_selfconsistency.py

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/processed/hotpot_clean.jsonl"
OUTPUT_CSV = "outputs/self_consistency_groq.csv"
PROMPT_TEMPLATE = open("prompts/cot.txt").read()

MODEL_NAME = "llama-3.3-70b-versatile"
NUM_SAMPLES = 5  # number of CoT samples per question

def parse_confidence(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last = lines[-1]
    m = re.match(r"^0(\.\d+)?$|^1(\.0+)?$", last)
    return float(last) if m else None

def parse_answer(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-2] if len(lines) >= 2 else lines[-1]

def main():
    df = pd.read_json(INPUT_FILE, lines=True).head(20)
    os.makedirs("outputs", exist_ok=True)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = PROMPT_TEMPLATE.format(
            context=row["context"],
            question=row["question"]
        )

        answers = []
        confidences = []

        for _ in range(NUM_SAMPLES):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0  # exploration
            )

            text = response.choices[0].message.content
            answers.append(parse_answer(text))
            confidences.append(parse_confidence(text) or 0.5)

        # majority vote
        pred = Counter(answers).most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences)

        rows.append({
            "question": row["question"],
            "gold": row["answer"],
            "pred": pred,
            "confidence": avg_conf,
            "samples": answers
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print("Saved ->", OUTPUT_CSV)

if __name__ == "__main__":
    main()
