# src/inference_groq.py

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import re
from tqdm import tqdm

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/processed/hotpot_clean.jsonl"
OUTPUT_CSV = "outputs/baseline_groq.csv"
PROMPT_TEMPLATE = open("prompts/baseline.txt").read()


# MODEL_NAME = "llama-3.1-8b-instant"
MODEL_NAME = "llama-3.3-70b-versatile"

def parse_confidence(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last = lines[-1]
    m = re.match(r"^0(\.\d+)?$|^1(\.0+)?$", last)
    return float(last) if m else None

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_json(INPUT_FILE, lines=True)
    df = df.head(20)  # test on 20 examples first

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        context = row["context"]
        question = row["question"]
        gold = row["answer"]

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content

        conf = parse_confidence(text)
        lines = text.split("\n")
        pred = lines[-2].strip() if len(lines) >= 2 else lines[-1].strip()

        rows.append({
            "question": question,
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "raw_response": text
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print("Saved ->", OUTPUT_CSV)

if __name__ == "__main__":
    main()
