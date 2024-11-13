import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel

# Load the Wine dataset
wine_columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
    'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
    'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_df = pd.read_csv("./Dataset/wine.data", header=None, names=wine_columns)

# Define feature matrix and target variable
X = wine_df.drop(columns='Alcohol').to_numpy()  # Convert to numpy array for compatibility
y = wine_df['Alcohol'].to_numpy()               # Convert target to numpy array

# Define parameters for the models
lambdas = np.logspace(-4, 4, 50)   # Adjust based on experimentation
hidden_units = [1, 3, 5, 10, 20]    # Adjust based on experimentation
K1, K2 = 10, 10  # Two-level cross-validation

# Baseline model function
def baseline_model(y_train, y_test):
    y_pred = np.full(y_test.shape, y_train.mean())
    return mean_squared_error(y_test, y_pred) / len(y_test)  # Squared loss per observation

# Outer cross-validation
kf_outer = KFold(n_splits=K1, shuffle=True, random_state=42)
results = {
    'Outer Fold': [],
    'ANN h*': [], 'ANN E_test': [],
    'Linear λ*': [], 'Linear E_test': [],
    'Baseline E_test': []
}

# Collect errors for each parameter setting across folds for final plotting
all_lambda_errors = {lmbda: [[] for _ in range(K1)] for lmbda in lambdas}
all_hidden_unit_errors = {h: [[] for _ in range(K1)] for h in hidden_units}

# Outer cross-validation loop
for fold, (train_outer_idx, test_outer_idx) in enumerate(kf_outer.split(X), start=1):
    X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
    y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

    # Standardize the features
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)

    # Inner cross-validation for model selection
    kf_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    best_ann_mse, best_ann_h = float('inf'), None
    best_linear_mse, best_lambda = float('inf'), None

    # Model evaluation within inner folds
    for train_inner_idx, test_inner_idx in kf_inner.split(X_train_outer):
        X_train_inner, X_test_inner = X_train_outer[train_inner_idx], X_train_outer[test_inner_idx]
        y_train_inner, y_test_inner = y_train_outer[train_inner_idx], y_train_outer[test_inner_idx]

        # ANN model: testing different hidden units
        for h in hidden_units:
            ann_model = MLPRegressor(hidden_layer_sizes=(h,), max_iter=20000, random_state=42, tol=1e-4)
            ann_model.fit(X_train_inner, y_train_inner)
            ann_mse = mean_squared_error(y_test_inner, ann_model.predict(X_test_inner))
            all_hidden_unit_errors[h][fold - 1].append(ann_mse)  # Store error for each fold

            hidden_avg_error = np.mean(all_hidden_unit_errors[h][fold - 1])
            if hidden_avg_error < best_ann_mse:
                best_ann_mse = hidden_avg_error
                best_ann_h = h

            #if ann_mse < best_ann_mse:
                #best_ann_mse, best_ann_h = ann_mse, h

        # Regularized linear regression: testing different lambdas
        for lmbda in lambdas:
            linear_model = Ridge(alpha=lmbda)
            linear_model.fit(X_train_inner, y_train_inner)
            linear_mse = mean_squared_error(y_test_inner, linear_model.predict(X_test_inner))
            all_lambda_errors[lmbda][fold - 1].append(linear_mse)  # Store error for each fold

            lambda_avg_error = np.mean(all_lambda_errors[lmbda][fold - 1])
            if lambda_avg_error < best_linear_mse:
                best_linear_mse = lambda_avg_error
                best_lambda = lmbda

            #if linear_mse < best_linear_mse:
                #best_linear_mse, best_lambda = linear_mse, lmbda

    # Outer test fold evaluation with selected parameters
    ann_model_final = MLPRegressor(hidden_layer_sizes=(best_ann_h,), max_iter=20000, random_state=42, tol=1e-4)
    ann_model_final.fit(X_train_outer, y_train_outer)
    ann_test_mse = mean_squared_error(y_test_outer, ann_model_final.predict(X_test_outer))

    linear_model_final = Ridge(alpha=best_lambda)
    linear_model_final.fit(X_train_outer, y_train_outer)
    linear_test_mse = mean_squared_error(y_test_outer, linear_model_final.predict(X_test_outer))

    baseline_test_mse = baseline_model(y_train_outer, y_test_outer)

    # Store results
    results['Outer Fold'].append(fold)
    results['ANN h*'].append(best_ann_h)
    results['ANN E_test'].append(ann_test_mse)
    results['Linear λ*'].append(best_lambda)
    results['Linear E_test'].append(linear_test_mse)
    results['Baseline E_test'].append(baseline_test_mse)

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("Two-Level Cross-Validation Results")
print(results_df)

# Plotting: Validation error vs Lambda (Linear Regression) for each fold in a single plot
plt.figure(figsize=(12, 5))

# Plot for lambda in Linear Regression
plt.subplot(1, 2, 1)
for fold in range(K1):
    fold_errors = {lmbda: np.mean(all_lambda_errors[lmbda][fold]) for lmbda in lambdas}
    plt.plot(fold_errors.keys(), fold_errors.values(), marker='o', label=f'Fold {fold + 1}')
plt.xlabel('Lambda')
plt.ylabel('Average Validation Error')
plt.title('Validation Error vs. Lambda (Linear Regression)')
plt.xscale('log')
plt.legend()

# Plot for hidden units in ANN
plt.subplot(1, 2, 2)
for fold in range(K1):
    fold_errors = {h: np.mean(all_hidden_unit_errors[h][fold]) for h in hidden_units}
    plt.plot(fold_errors.keys(), fold_errors.values(), marker='o', label=f'Fold {fold + 1}')
plt.xlabel('Hidden Units')
plt.ylabel('Average Validation Error')
plt.title('Validation Error vs. Hidden Units (ANN)')
plt.legend()

plt.tight_layout()
plt.show()

# Part 3: Statistical Evaluation using Paired t-tests
# Collect errors for each model across folds
lr_errors = results_df['Linear E_test']
ann_errors = results_df['ANN E_test']
baseline_errors = results_df['Baseline E_test']

# Paired t-tests
_, p_value_lr_ann = ttest_rel(lr_errors, ann_errors)
_, p_value_lr_baseline = ttest_rel(lr_errors, baseline_errors)
_, p_value_ann_baseline = ttest_rel(ann_errors, baseline_errors)

# Display paired t-test results
print("\nStatistical Evaluation using Paired t-tests:")
print(f"Linear Regression vs ANN: p-value = {p_value_lr_ann:.4f}")
print(f"Linear Regression vs Baseline: p-value = {p_value_lr_baseline:.4f}")
print(f"ANN vs Baseline: p-value = {p_value_ann_baseline:.4f}")
