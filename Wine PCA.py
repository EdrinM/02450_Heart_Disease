import numpy as np
#Dataset is the full data including the classification label in the first column
dataset = np.loadtxt(r"C:\Users\spill\Downloads\wine\wine.data", delimiter=',',dtype=float)

# X is the actual data without the classification label
X = (dataset[:,1:]).copy()
classlabel=(dataset[:,0]).copy()
X_standard = np.empty((178,13))
# standizing data
for column in range(13):
    X_standard[:,column] = X[:,column]
    feature_mean=(X[:,column]).mean()
    feature_standard_deviation=(X[:,column]).std()
    X_standard[:,column] = (X_standard[:,column]-feature_mean*(np.ones((178,1))[:,0]))/feature_standard_deviation

#preforming svd
from scipy.linalg import svd
U, S, V = svd(X_standard, full_matrices=False)

#understanding the value from each principal component (code copy from exercise 2_1_3)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rho = (S * S) / (S * S).sum()
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()
plt.close()

#Projecting the data into the PCA space
Z = X_standard @ V
for i in range(178):
    if classlabel[i]==1.:
        plt.plot(Z[i,0],Z[i,1],"go")
    elif classlabel[i]==2.:
        plt.plot(Z[i,0],Z[i,1],"ro")
    else:
        plt.plot(Z[i,0],Z[i,1],"bo")

plt.title("Wine origen: PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='winery 1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='winery 2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='winery 3')
]
# Add the legend to the plot
plt.legend(handles=legend_elements)
plt.show() #showing with 2 PC in 2D
plt.close()

#I will show it in 3D with 3 principal components
ax = plt.axes(projection ="3d")
for i in range(178):
    if classlabel[i]==1.:
        ax.scatter3D(Z[i,0], Z[i,1], Z[i,2], color = "green")
    elif classlabel[i]==2.:
        ax.scatter3D(Z[i,0], Z[i,1], Z[i,2], color = "red")
    else:
        ax.scatter3D(Z[i,0], Z[i,1], Z[i,2], color = "blue")

plt.title("Wine origen: PCA 3D")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='winery 1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='winery 2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='winery 3')
]
# Add the legend to the plot
plt.legend(handles=legend_elements)
plt.show() #showing with 3 PC in 3D
plt.close()





