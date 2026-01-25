import pandas as pd
import os

# 1. Define your experiment structure
VERSIONS = [
    "baseline", "baseline_multi", "cot_answer", "cot_answer_multi",
    "cot_confidence", "cot_confidence_multi", "cot_answer_confidence",
    "cot_answer_confidence_multi", "scientific"
]

# Map your folders to the specific model names used in filenames
MODEL_MAP = {
    "llama": "llama-3.1-8b-instant",
    "src_gemma": "gemma-3-4b-it",
    "src_gemma_27b": "gemma-3-27b-it"
}

summary_rows = []

# 2. Iterate through every file
for folder, model_name in MODEL_MAP.items():
    for version in VERSIONS:
        file_path = f"{folder}/{version}_{model_name}_metrics.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Ensure columns are numeric
            df['f1'] = pd.to_numeric(df['f1'], errors='coerce')
            df['bert'] = pd.to_numeric(df['bert'], errors='coerce')
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

            # --- FILTERING STEP ---
            # Remove rows where confidence, f1, or bert are > 1
            # (which causes those massive MSE outliers in your previous data)
            initial_count = len(df)
            df = df[
                (df['confidence'] >= 0.0) & (df['confidence'] <= 1.0) &
                (df['f1'] >= 0.0) & (df['f1'] <= 1.0) &
                (df['bert'] >= 0.0) & (df['bert'] <= 1.0)
                ].copy()

            filtered_count = len(df)
            if initial_count != filtered_count:
                print(f"Filtered {initial_count - filtered_count} rows from {file_path}")

            # Calculate MSE (Squared Error) per individual prompt on CLEAN data
            if not df.empty:
                mse_f1_scores = (df['confidence'] - df['f1']) ** 2
                mse_bert_scores = (df['confidence'] - df['bert']) ** 2

                # Calculate the average across all valid prompts in this version
                summary_rows.append({
                    'Model': model_name,
                    'Version': version,
                    'average mse f1': mse_f1_scores.mean(),
                    'average mse bert': mse_bert_scores.mean()
                })
            else:
                print(f"Warning: No valid rows left in {file_path} after filtering.")
        else:
            print(f"Skipping missing file: {file_path}")

# 3. Save the results
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("model_ablation_mse_summary.csv", index=False)

print("\n--- Summary CSV Created Successfully ---")
print(summary_df.head())