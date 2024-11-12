import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess data
dataset = np.loadtxt(r".\Dataset\wine.data", delimiter=',', dtype=float)
X = dataset[:, [0] + list(range(2, dataset.shape[1]))].copy()   # Everything except alcohol
y = dataset[:, 1].copy()    # Alcohol is the target variable


# Task 1 - Baseline model (predict mean of y)
def baseline_model(y_train):
    mean_y = np.mean(y_train)
    return lambda X: np.full(X.shape[0], mean_y)


# Task 1 - Set hyperparameter ranges
lambdas = np.logspace(-4, 4, 10)  # Adjust based on experimentation
hidden_units = [1, 5, 10, 15, 20]  # Adjust based on experimentation

# Task 1 - Initialize KFold with K1 = K2 = 10 for outer and inner loops
K1 = 10
K2 = 10
outer_kf = KFold(n_splits=K1, shuffle=True, random_state=42)

# Task 2 - Store results for each fold in a table format
results_table = []

# Task 1 - Outer cross-validation loop
for train_idx, test_idx in outer_kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Task 1 - Inner cross-validation loop for hyperparameter tuning
    best_lambda = None
    best_hidden_units = None
    best_lr_error = float('inf')
    best_ann_error = float('inf')

    inner_kf = KFold(n_splits=K2, shuffle=True, random_state=42)
    for lam in lambdas:
        # Task 1 - Linear regression with regularization (Ridge regression)
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y[inner_train_idx], y[inner_val_idx]

            model = Ridge(alpha=lam)
            model.fit(X_inner_train, y_inner_train)
            y_pred = model.predict(X_inner_val)
            inner_errors.append(mean_squared_error(y_inner_val, y_pred))

        avg_error = np.mean(inner_errors)
        if avg_error < best_lr_error:
            best_lr_error = avg_error
            best_lambda = lam

    # Task 1 - ANN with different hidden units
    for h in hidden_units:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y[inner_train_idx], y[inner_val_idx]

            model = MLPRegressor(hidden_layer_sizes=(h,), max_iter=1000, random_state=42)
            model.fit(X_inner_train, y_inner_train)
            y_pred = model.predict(X_inner_val)
            inner_errors.append(mean_squared_error(y_inner_val, y_pred))

        avg_error = np.mean(inner_errors)
        if avg_error < best_ann_error:
            best_ann_error = avg_error
            best_hidden_units = h

    # Task 2 - Evaluate on the test set with the best lambda and h
    # Linear regression
    final_lr_model = Ridge(alpha=best_lambda)
    final_lr_model.fit(X_train, y_train)
    y_pred_lr = final_lr_model.predict(X_test)
    lr_test_error = mean_squared_error(y_test, y_pred_lr)

    # ANN
    final_ann_model = MLPRegressor(hidden_layer_sizes=(best_hidden_units,), max_iter=1000, random_state=42)
    final_ann_model.fit(X_train, y_train)
    y_pred_ann = final_ann_model.predict(X_test)
    ann_test_error = mean_squared_error(y_test, y_pred_ann)

    # Baseline
    baseline_predictor = baseline_model(y_train)
    y_pred_baseline = baseline_predictor(X_test)
    baseline_test_error = mean_squared_error(y_test, y_pred_baseline)

    # Task 2 - Store results in the table format (like Table 1)
    results_table.append({
        'Fold': len(results_table) + 1,
        'Best_lambda': best_lambda,
        'Best_hidden_units': best_hidden_units,
        'LR_test_error': lr_test_error,
        'ANN_test_error': ann_test_error,
        'Baseline_test_error': baseline_test_error
    })

# Task 2 - Print table of results
print("Fold | Best λ | Best h | Linear Regression Test Error | ANN Test Error | Baseline Test Error")
for result in results_table:
    print(f"{result['Fold']:>4} | {result['Best_lambda']:.4f} | {result['Best_hidden_units']:>4} | "
          f"{result['LR_test_error']:.4f} | {result['ANN_test_error']:.4f} | {result['Baseline_test_error']:.4f}")

# Task 3 - Statistical evaluation using paired t-test
from scipy.stats import ttest_rel

# Collect errors for each model across folds
lr_errors = [result['LR_test_error'] for result in results_table]
ann_errors = [result['ANN_test_error'] for result in results_table]
baseline_errors = [result['Baseline_test_error'] for result in results_table]

# Paired t-tests
_, p_value_lr_ann = ttest_rel(lr_errors, ann_errors)
_, p_value_lr_baseline = ttest_rel(lr_errors, baseline_errors)
_, p_value_ann_baseline = ttest_rel(ann_errors, baseline_errors)

print("\nTask 3 - Paired t-test results:")
print(f"Linear Regression vs ANN: p-value = {p_value_lr_ann:.4f}")
print(f"Linear Regression vs Baseline: p-value = {p_value_lr_baseline:.4f}")
print(f"ANN vs Baseline: p-value = {p_value_ann_baseline:.4f}")
