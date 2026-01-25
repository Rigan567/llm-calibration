import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


if len(sys.argv) < 2:
    print("Usage: python inference_groq_com.py <prompt_version>")
    sys.exit(1)

PROMPT_VERSION = sys.argv[1]
FOLDER = sys.argv[2] #"llama"
MODEL_NAME = sys.argv[3] #llama-3.1-8b-instant
INPUT_FILE = f"{FOLDER}/{PROMPT_VERSION}_{MODEL_NAME}_metrics.csv"
OUTPUT_CSV = f"plots/{FOLDER}/{PROMPT_VERSION}_{MODEL_NAME}_accuracy_vs_confidence.png"

# 1. Load your data
df = pd.read_csv(INPUT_FILE)

# 1. Force confidence to be a number (float)
df['confidence'] = pd.to_numeric(df['confidence'])

plt.figure(figsize=(10, 6))

# 2. Use a Scatterplot instead of Stripplot for continuous data
# Adding 's' for size and 'alpha' for transparency helps if points overlap
sns.scatterplot(data=df, y='confidence', x='f1', s=25, color='teal', alpha=0.6)

# 3. Draw the calibration line strictly from 0 to 1
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Ideal Calibration')

# 4. Set explicit limits for the axis so it doesn't "zoom in" too much
plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.title('Confidence vs. Accuracy (F1)')
plt.ylabel('Model Confidence Score')
plt.xlabel('Actual Accuracy (F1)')
plt.legend(['Individual Prompts', 'Ideal Calibration'])

plt.savefig(OUTPUT_CSV, dpi=300, bbox_inches='tight')
#plt.show()