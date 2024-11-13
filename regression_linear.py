import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge


# Load the dataset into a pandas DataFrame
columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
           "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
           "OD280/OD315 of diluted wines", "Proline"]

# Task 1:
# ----------------------------------------------------------------------------------------------------------------------
dataset = np.loadtxt(r".\Dataset\wine.data", delimiter=',',dtype=float)
X = dataset[:, [0] + list(range(2, dataset.shape[1]))].copy()       # Everything except alcohol
y = (dataset[:, 1]).copy()                                          # Alcohol is the target variable

# Feature transformation (mean 0, std deviation 13 for each column)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Task 2:
# ----------------------------------------------------------------------------------------------------------------------
# Regularization with lambda (using Ridge regression with cross-validation)
lambdas = np.logspace(-4, 4, 50)    # Range of λ values from very small to large
cv_errors = []                      # Store cross-validation errors for each λ

kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation

for lam in lambdas:
    model = Ridge(alpha=lam)
    # Cross-validation with negative MSE as score
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    cv_errors.append(-scores.mean())  # Store mean CV error for current λ

'''
cv_errors = []
for lam in lambdas:
    model = Ridge(alpha=lam)
    fold_errors = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_errors.append(mean_squared_error(y_test, y_pred))

    cv_errors.append(np.mean(fold_errors))
'''

# Plot the cross-validated generalization error as a function of λ
plt.figure(figsize=(10, 6))
plt.plot(lambdas, cv_errors, marker='o')
plt.xscale('log')
plt.xlabel('Regularization parameter λ')
plt.ylabel('Estimated Generalization Error (MSE)')
plt.title('Generalization Error as a Function of λ')
plt.show()

# Task 3:
# ----------------------------------------------------------------------------------------------------------------------

# Selecting the model with the lowest generalization error
best_lambda = lambdas[np.argmin(cv_errors)]
best_model = Ridge(alpha=best_lambda)
best_model.fit(X, y)

# Explanation of the effect of individual attributes on y
# The effect of each attribute is captured by the coefficient in the best model
for i, coef in enumerate(best_model.coef_):
    print(f"Feature {i} has a coefficient of {coef:.3f}")

# Predicting y for a new input x
example_x = X[0]  # Take an example input from X
example_x = scaler.transform([example_x]) * 13
predicted_y = best_model.predict(example_x)

print("Predicted y for example input:", predicted_y)
