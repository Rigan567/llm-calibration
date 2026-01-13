import pandas as pd

# Load the dataset
df = pd.read_json("combined_qa_dataset_800.jsonl", lines=True)

# Check for duplicates based on 'question' and 'answer' columns
duplicates = df.duplicated(subset=["question", "answer"], keep=False)

# Print summary
num_duplicates = duplicates.sum()
print(f"Number of duplicate QA pairs: {num_duplicates}")

# Optionally, show the duplicates
if num_duplicates > 0:
    print("Duplicate entries:")
    print(df[duplicates])
else:
    print("No duplicates found.")
