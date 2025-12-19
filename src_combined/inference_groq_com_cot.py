# src/inference_groq_com_cot.py

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "data/processed/combined_clean.jsonl"
OUTPUT_CSV = "outputs/baseline_groq_cot.csv"

# Chain-of-Thought style prompt
COT_PROMPT_TEMPLATE = """
You are a scientific fact verification assistant.

You will decide whether the following statement is factually true or false.
First, think step by step and reason briefly.
Then, on the LAST TWO LINES, output ONLY:

yes or no
a confidence score between 0 and 1

Do NOT add labels like "Answer" or "Confidence".
Do NOT add any other text after the confidence.

Statement: {question}
"""


MODEL_NAME = "llama-3.1-8b-instant"   # you can swap to a stronger model if you want


def parse_model_output(text: str):
    """
    Parse model output of the form:

    <reasoning...>
    yes
    0.87

    Robust to minor formatting issues.
    """
    txt = text.strip()
    lines = [l.strip().lower() for l in txt.split("\n") if l.strip()]

    # --- extract yes/no / true/false ---
    ans = None

    # check lines from bottom upwards (answer should be near the end)
    for l in reversed(lines):
        if l in ["yes", "no", "true", "false"]:
            if l in ["yes", "true"]:
                ans = "yes"
            else:
                ans = "no"
            break

    # fallback: look anywhere in text
    if ans is None:
        low = txt.lower()
        if " yes" in " " + low or low.startswith("yes"):
            ans = "yes"
        elif " no" in " " + low or low.startswith("no"):
            ans = "no"
        elif "true" in low:
            ans = "yes"
        elif "false" in low:
            ans = "no"
        else:
            ans = "yes"   # default fallback

    # --- extract confidence (first number between 0 and 1) ---
    conf = None
    low = txt.lower()
    numbers = re.findall(r"\d*\.\d+|\d+", low)  # grab all numbers

    for num in numbers:
        try:
            v = float(num)
            if 0 <= v <= 1:
                conf = v
                break
        except ValueError:
            continue

    if conf is None:
        conf = 0.5

    return ans, conf, text


def main():
    df = pd.read_json(INPUT_FILE, lines=True)

    # you can change 100 to a larger number if you want
    df = df.head(20)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        gold = row["answer"]

        prompt = COT_PROMPT_TEMPLATE.format(question=question)

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

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
