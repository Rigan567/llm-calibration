import matplotlib.pyplot as plt
import numpy as np

methods = ["Baseline", "CoT", "Self-Consistency"]
accuracy = [0.20, 0.10, 0.10]
brier = [0.557, 0.773, 0.737]
ece = [0.09, 0.365, 0.773]

x = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, accuracy, width, label="Accuracy")
plt.bar(x, brier, width, label="Brier Score")
plt.bar(x + width, ece, width, label="ECE")

plt.xticks(x, methods)
plt.ylabel("Score")
plt.title("Baseline vs CoT vs Self-Consistency")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
