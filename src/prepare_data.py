# src/prepare_data.py

from datasets import load_dataset, DownloadConfig
import pandas as pd
import os
import shutil
from pathlib import Path

def get_hf_cache_dir():
    hf_cache = os.getenv("HF_DATASETS_CACHE")
    if hf_cache:
        return Path(hf_cache)
    return Path.home() / ".cache" / "huggingface" / "datasets"

def load_and_save(dataset_name, save_dir):
    print(f"Loading dataset: {dataset_name} ...")

    try:
        if dataset_name == "hotpotqa/hotpot_qa":
            ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
        else:
            ds = load_dataset(dataset_name)
    except OSError as e:
        if "Consistency check failed" in str(e):
            print("⚠ Consistency check failed. Clearing cache and retrying...")

            cache_path = get_hf_cache_dir()
            if cache_path.exists():
                print(f"Deleting cache at {cache_path} ...")
                shutil.rmtree(cache_path)

            ds = load_dataset("hotpotqa/hotpot_qa", "distractor", 
                              download_config=DownloadConfig(force_download=True))
        else:
            raise

    os.makedirs(save_dir, exist_ok=True)

    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        save_path = os.path.join(save_dir, f"{split}.jsonl")
        df.to_json(save_path, orient="records", lines=True)
        print(f"Saved {split} -> {save_path}")

    print("✅ Done!\n")

if __name__ == "__main__":
    load_and_save("hotpotqa/hotpot_qa", "data/raw/hotpotqa")
