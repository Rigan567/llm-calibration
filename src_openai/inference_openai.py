import os
import sys
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

if len(sys.argv) < 2:
    print("Usage: python inference_gemma.py <prompt_version>")
    sys.exit(1)

PROMPT_VERSION = sys.argv[1]

try:
    with open("Groq_api_key.txt", "r") as f:
        key = f.read().strip()
except FileNotFoundError:
    key = os.getenv("GROQ_API_KEY")

if not key:
    raise RuntimeError("Groq API Key not found.")

client = Groq(api_key=key)

INPUT_FILE = "combined_qa_dataset_800.jsonl"
MODEL_NAME = "openai/gpt-oss-120b"
OUTPUT_CSV = f"outputs/{PROMPT_VERSION}_{MODEL_NAME.replace('/', '_')}.csv"

TARGET_SOURCES = {
    "Astro-QA_Judgement",
    "HotpotQA",
    "GlobalMedQA_EN",
    "TORQUE"
}

def parse_model_output(text):
    lines = [line for line in text.splitlines() if line.strip() != '']
    if not lines:
        return "N/A", None, text

    if len(lines) == 1:
        answer = lines[0]
        confidence = None
    else:
        answer = lines[-2]
        try:
            confidence = float(lines[-1].strip())
        except (ValueError, IndexError):
            confidence = None
    return answer, confidence, text

def main():
    df = pd.read_json(INPUT_FILE, lines=True)

    if "source" not in df.columns:
        raise ValueError("'source' column not found.")

    df = df[df["source"].isin(TARGET_SOURCES)]

    processed_questions = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            processed_questions = set(existing_df['question'].astype(str).tolist())
            print(f"🔄 Resuming {PROMPT_VERSION}: Skipping {len(processed_questions)} samples already in CSV.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")

    os.makedirs("outputs", exist_ok=True)

    remaining_df = df[~df['question'].astype(str).isin(processed_questions)]
    print(f"Total samples to process: {len(remaining_df)}")
    print(remaining_df["source"].value_counts())

    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):

        try:
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

            prompt = prompt_template.format(question=row["question"])

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )

            text = response.choices[0].message.content.strip()
            pred, conf, raw = parse_model_output(text)

            result_row = pd.DataFrame([{
                "question": row["question"],
                "gold": row["answer"],
                "pred": pred,
                "confidence": conf,
                "source": row["source"],
                "raw_response": raw
            }])

            result_row.to_csv(
                OUTPUT_CSV,
                mode='a',
                index=False,
                header=not os.path.exists(OUTPUT_CSV)
            )

        except Exception as e:
            err_msg = str(e).lower()
            # If we hit the 429 Rate Limit (TPD/RPM) or Quota, we stop the whole script
            if "rate_limit" in err_msg or "quota" in err_msg or "429" in err_msg:
                print(f"\n🛑 QUOTA EXCEEDED: Groq limits reached. Stopping. Error: {e}")
                break

            print(f"Error on question: {e}")
            continue


print(f"\n✅ Saved results -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
