# src/inference_groq_com.py

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()

file = open("Groq_api_key.txt", "r")
key = file.read()
client = Groq(api_key=key)

INPUT_FILE = "data/combined_qa_dataset_800.jsonl"
OUTPUT_CSV = "outputs/baseline_groq.csv"
PROMPT_TEMPLATE = open("prompts/baseline.txt").read()

# MODEL_NAME = "llama-3.1-8b-instant"
MODEL_NAME = "llama-3.3-70b-versatile"


def parse_model_output(text):
    txt = text.lower().strip()

    # detect yes/no anywhere in text
    if "yes" in txt:
        ans = "yes"
    elif "no" in txt:
        ans = "no"
    elif "true" in txt:
        ans = "yes"
    elif "false" in txt:
        ans = "no"
    else:
        ans = "yes"  # fallback

    # extract first float between 0 and 1
    numbers = re.findall(r"0\.\d+|1\.0+|1|0", txt)

    conf = None
    for num in numbers:
        try:
            v = float(num)
            if 0 <= v <= 1:
                conf = v
                break
        except:
            pass

    if conf is None:
        conf = 0.5

    return ans, conf, text


def main():
    df = pd.read_json(INPUT_FILE, lines=True)
    df = df.head(500)  # small evaluation batch

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        gold = row["answer"]

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
            "raw_response": raw
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
