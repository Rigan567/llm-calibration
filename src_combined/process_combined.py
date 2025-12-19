import json
import os

RAW_FILE = "data/raw/combined/combined_qa_ds.jsonl"
OUT_FILE = "data/processed/combined_clean.jsonl"

def normalize_answer(ans):
    ans = ans.strip().lower()
    if ans in ["true", "yes"]:
        return "yes"
    if ans in ["false", "no"]:
        return "no"
    return ans

def main():
    os.makedirs("data/processed", exist_ok=True)

    with open(RAW_FILE, "r", encoding="utf8") as fin, \
         open(OUT_FILE, "w", encoding="utf8") as fout:

        for idx, line in enumerate(fin):
            obj = json.loads(line)

            question = obj.get("question", "")
            answer = normalize_answer(obj.get("answer", ""))

            out = {
                "id": idx,
                "question": question,
                "context": "",   # no context available
                "answer": answer
            }

            fout.write(json.dumps(out) + "\n")

    print(f"Saved processed dataset -> {OUT_FILE}")

if __name__ == "__main__":
    main()
