import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset into a pandas DataFrame
columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", 
           "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", 
           "OD280/OD315 of diluted wines", "Proline"]
dataset = np.loadtxt(r'.\Dataset\wine.data', delimiter=',', dtype=float)
wine_df = pd.DataFrame(dataset, columns=columns)

# Calculate correlation matrix
correlation_matrix = wine_df.corr()

# Heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix for Wine Dataset")
plt.show()
