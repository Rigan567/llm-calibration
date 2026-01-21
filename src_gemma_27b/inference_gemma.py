import os
import sys
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DELAY_BETWEEN_REQUESTS = 2.1

if len(sys.argv) < 2:
    print("Usage: python inference_gemma.py <prompt_version>")
    sys.exit(1)

PROMPT_VERSION = sys.argv[1]

try:
    with open("Gemma_api_key.txt", "r") as f:
        key = f.read().strip()
except FileNotFoundError:
    key = os.getenv("GEMMA_API_KEY")

if not key:
    raise RuntimeError("GEMMA_API_KEY Key not found.")

genai.configure(api_key=key)

INPUT_FILE = "combined_qa_dataset_800.jsonl"
MODEL_NAME = "gemma-3-27b-it"
OUTPUT_CSV = f"outputs/{PROMPT_VERSION}_{MODEL_NAME}.csv"
model = genai.GenerativeModel(MODEL_NAME)

TARGET_SOURCES = {
    "Astro-QA_Judgement",   # True / False
    "Astro-QA_Subjective",
    "HotpotQA",             # Multi-hop factoid
    "GlobalMedQA_EN",       # Medical MCQ
    "TORQUE"            # Temporal reasoning (if present)
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
        raise ValueError(
            "'source' column not found. "
        )

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
        start_time = time.time()
        try:
            type = {
                "Open ended": "open",
                "True or False": "true_false",
                "Multiple-choice": "choice",
                "temporal": "temporal",
            }.get(row["type"])
            prompt_template = open(BASE_DIR/f"prompts_gemma/{PROMPT_VERSION}/{PROMPT_VERSION}_{type}.txt").read()
            prompt = prompt_template.format(question=row["question"])
            response = model.generate_content(prompt)
            text = response.text.strip()

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
            """
            err_msg = str(e).lower()
            # If we hit the 429 Rate Limit (TPD/RPM) or Quota, we stop the whole script
            if "rate_limit" in err_msg or "quota" in err_msg or "429" in err_msg:
                print(f"\n🛑 QUOTA EXCEEDED: Groq limits reached. Stopping. Error: {e}")
                break

            print(f"Error on question: {e}")
            continue
            """

        elapsed_time = time.time() - start_time
        if elapsed_time < DELAY_BETWEEN_REQUESTS:
            time.sleep(DELAY_BETWEEN_REQUESTS - elapsed_time)


print(f"\n✅ Saved results -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
