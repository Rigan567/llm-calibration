"""
Process raw HotpotQA JSONL into a simple, consistent JSONL for LLM prompting.
Output fields: id, question, answer, context, supporting_facts (optional)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

RAW_PATH = Path("data/raw/hotpotqa/validation.jsonl")
OUT_PATH = Path("data/processed/hotpot_clean.jsonl")

def extract_answer(record: Dict[str, Any]):
    # HotpotQA typically has "answer" as a string
    if "answer" in record and record["answer"]:
        return record["answer"]
    # fallback: answers or some other key
    if "answers" in record and record["answers"]:
        if isinstance(record["answers"], (list, tuple)):
            return record["answers"][0]
        if isinstance(record["answers"], dict) and "text" in record["answers"]:
            return record["answers"]["text"]
    return ""

def extract_question(record: Dict[str, Any]):
    return record.get("question") or record.get("query") or ""

def extract_supporting_facts(record: Dict[str, Any]) -> List[str]:
    # Hotpot often has "supporting_facts": list of [title, sentence_idx] pairs
    sf = record.get("supporting_facts") or record.get("supporting_facts_context") or []
    out = []
    try:
        for item in sf:
            # If item is [title, idx]
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                out.append(item[0])
            elif isinstance(item, dict):
                out.append(item.get("title") or str(item))
            else:
                out.append(str(item))
    except Exception:
        pass
    return out

def extract_context(record: Dict[str, Any]) -> str:
    # Various key names possible: "context", "context_paragraphs", "context_text"
    # In HuggingFace Hotpot, there is often a "context" field which is a list of [title, paragraph].
    ctx = record.get("context") or record.get("contexts") or record.get("context_text") or []
    if isinstance(ctx, list):
        # If list of [title, paragraph] pairs
        parts = []
        for item in ctx:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                paragraph = item[1]
                parts.append(f"{title}: {paragraph}")
            elif isinstance(item, dict):
                # dict with keys maybe 'title' and 'text'
                title = item.get("title", "")
                paragraph = item.get("text", "") or item.get("paragraph", "")
                parts.append(f"{title}: {paragraph}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    elif isinstance(ctx, str):
        return ctx
    else:
        return ""

def process():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with RAW_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            q = extract_question(obj)
            a = extract_answer(obj)
            context = extract_context(obj)
            supporting = extract_supporting_facts(obj)
            out = {
                "id": obj.get("id") or str(count),
                "question": q,
                "answer": a,
                "context": context,
                "supporting_facts": supporting
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    print(f"Processed {count} examples -> {OUT_PATH}")

if __name__ == "__main__":
    if not RAW_PATH.exists():
        print(f"Raw file not found: {RAW_PATH}")
    else:
        process()
