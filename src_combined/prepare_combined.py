import json
import os
from sklearn.model_selection import train_test_split

RAW_FILE = "data/raw/combined/combined_qa_ds.jsonl"
OUT_DIR = "data/raw/combined_split"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load lines
    data = []
    with open(RAW_FILE, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} examples")

    # Split into train/validation
    train, val = train_test_split(data, test_size=0.2, random_state=42)

    # Save them
    with open(f"{OUT_DIR}/train.jsonl", "w", encoding="utf8") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    with open(f"{OUT_DIR}/validation.jsonl", "w", encoding="utf8") as f:
        for row in val:
            f.write(json.dumps(row) + "\n")

    print("Saved train/validation split!")

if __name__ == "__main__":
    main()
