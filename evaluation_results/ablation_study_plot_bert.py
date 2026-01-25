import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load your actual data
df_raw = pd.read_csv('model_ablation_mse_summary.csv')

#MODEL="llama-3.1-8b-instant"
MODEL="gemma-3-4b-it"
#MODEL="gemma-3-27b-it"
OUTPUT_CSV = f"{MODEL}_ablation_study_mse_bert.png"

df_llama = df_raw[df_raw['Model'] == MODEL].copy()

# 2. Data Cleaning & Mapping
# Filter out the impossible outliers (> 1.0) so they don't ruin the plot
# (Note: You should investigate your CSVs to see why these numbers are so high!)
df_clean = df_raw[df_raw['average mse bert'] <= 1.0].copy()

# Map your Version names to Plot Categories
version_map = {
    'baseline': ('Baseline', 'Standard'),
    'baseline_multi': ('Baseline', '+ Few-Shot'),
    'cot_answer': ('CoT (Answer)', 'Standard'),
    'cot_answer_multi': ('CoT (Answer)', '+ Few-Shot'),
    'cot_confidence': ('CoT (Confidence)', 'Standard'),
    'cot_confidence_multi': ('CoT (Confidence)', '+ Few-Shot'),
    'cot_answer_confidence': ('CoT (Both)', 'Standard'),
    'cot_answer_confidence_multi': ('CoT (Both)', '+ Few-Shot'),
    'scientific': ('Scientific', 'Standard')
}

# Apply the mapping
df_clean['Technique'] = df_clean['Version'].map(lambda x: version_map.get(x, (x, 'Standard'))[0])
df_clean['Setting'] = df_clean['Version'].map(lambda x: version_map.get(x, (x, 'Standard'))[1])

# 3. Visualization Styling
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = {"Standard": "#5b84c1", "+ Few-Shot": "#f2b035"}

# 4. Create the Bar Plot (using average mse bert)
ax = sns.barplot(
    data=df_clean,
    x='Technique',
    y='average mse bert',
    hue='Setting',
    palette=palette,
    edgecolor='black',
    linewidth=1.2,
    order=['Baseline', 'CoT (Answer)', 'CoT (Confidence)', 'CoT (Both)', 'Scientific']
)

# 5. Aesthetics
plt.title('Ablation Study: Average MSE BERT Score(Calibration Error)', fontsize=18, fontweight='bold', pad=25)
plt.ylabel('Mean Squared Error (Lower is Better)', fontsize=14)
plt.xlabel('')
plt.ylim(0, 0.2) # MSE shouldn't exceed 1.0
plt.legend(title='Condition', loc='upper right', frameon=True, shadow=True)

# Add numeric labels on top
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_CSV, dpi=300)
plt.show()

print(f"Plot generated. Note: {len(df_raw) - len(df_clean)} outliers were removed.")