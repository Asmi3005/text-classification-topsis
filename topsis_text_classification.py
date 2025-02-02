import numpy as np
from topsis import topsis

# Example decision matrix
decision_matrix = np.array([
    [90, 88, 85, 87, 500, 450],  # BERT
    [92, 89, 87, 88, 700, 500],  # RoBERTa
    [87, 85, 83, 84, 300, 250],  # DistilBERT
    [91, 87, 86, 87, 600, 480],  # XLNet
    [89, 86, 84, 85, 400, 300]   # ALBERT
])

weights = np.array([0.3, 0.2, 0.2, 0.2, 0.05, 0.05])
beneficial = [True, True, True, True, False, False]

# Run TOPSIS
topsis_scores, rankings = topsis(decision_matrix, weights, beneficial)

# Normalize TOPSIS scores between 0 and 1
min_score = min(topsis_scores)
max_score = max(topsis_scores)
normalized_scores = [(score - min_score) / (max_score - min_score) for score in topsis_scores]

# Rank models based on normalized scores (1-based)
sorted_indices = np.argsort(normalized_scores)[::-1]
ranks = [i + 1 for i in sorted_indices]

# Display results
models = ["BERT", "RoBERTa", "DistilBERT", "XLNet", "ALBERT"]
for i, model in enumerate(models):
    print(f"{model}: Normalized Score = {normalized_scores[i]:.4f}, Rank = {ranks[i]}")


import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the TOPSIS Scores using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=topsis_scores, palette='viridis')

# Add titles and labels
plt.title('TOPSIS Scores for Each Model', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('TOPSIS Score', fontsize=14)

# Show plot
plt.show()

