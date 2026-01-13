# src/process_combined.py

import json
from pathlib import Path

RAW_PATH = Path("data/raw/combined/combined_qa_ds.jsonl")
OUT_PATH = Path("data/processed/combined_clean.jsonl")

def normalize_answer(ans):
    """
    Light normalization ONLY.
    Do NOT force everything to yes/no.
    """
    if ans is None:
        return ""

    ans = str(ans).strip()

    # normalize common booleans
    if ans.lower() in ["true", "yes"]:
        return "yes"
    if ans.lower() in ["false", "no"]:
        return "no"

    return ans


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0

    with RAW_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)

            question = obj.get("question", "").strip()
            answer = normalize_answer(obj.get("answer", ""))
            source = obj.get("source", "unknown")

            if question == "":
                continue

            rows.append({
                "id": idx,
                "question": question,
                "answer": answer,
                "source": source   # ✅ KEEP SOURCE
            })

            idx += 1

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
