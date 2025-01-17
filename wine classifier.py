import numpy as np
import sklearn.linear_model as lm
import sklearn.neighbors as Knb
import sklearn.dummy as dum
from matplotlib.pylab import figure, legend, show, xlabel, ylabel, semilogx, plot
from sklearn import model_selection

#Dataset is the full data including the classification label in the first column
dataset = np.loadtxt(r".\Dataset\wine.data", delimiter=',',dtype=float)
X = (dataset[:,1:]).copy() # the chemical attributes of the wines
y=(dataset[:,0]).copy() # the winery that produced the given wine

# Outer CV settings
outer_K = 10
outer_cv = model_selection.KFold(n_splits=outer_K, shuffle=True)

# Inner CV settingss
inner_K = 20

# Define range of regularization parameters
lambda_interval = np.logspace(-3, 2, 45)
outer_errors = []

for outer_k, (train_index, test_index) in enumerate(outer_cv.split(X)):
    print(f"Computing outer CV fold: {outer_k + 1}/{outer_K}...")

    # Split data into outer training and test sets
    X_train_outer, y_train_outer = X[train_index, :], y[train_index]
    X_test_outer, y_test_outer = X[test_index, :], y[test_index]

    # Standardize data based on outer training set
    mu, sigma = np.mean(X_train_outer, 0), np.std(X_train_outer, 0)
    X_train_outer = (X_train_outer - mu) / sigma
    X_test_outer = (X_test_outer - mu) / sigma

    # Inner cross-validation for hyperparameter tuning
    inner_cv = model_selection.KFold(n_splits=inner_K, shuffle=True)
    inner_errors_test = np.empty(len(lambda_interval))
    inner_errors_train = np.empty(len(lambda_interval))

    for i, lambda_val in enumerate(lambda_interval):
        inner_fold_errors = []
        inner_fold_errors_train = []

        # Inner loop: 20-fold CV for tuning lambda
        for inner_train_index, inner_val_index in inner_cv.split(X_train_outer):
            X_train_inner, y_train_inner = X_train_outer[inner_train_index], y_train_outer[inner_train_index]
            X_val_inner, y_val_inner = X_train_outer[inner_val_index], y_train_outer[inner_val_index]

            # Train model with current lambda
            mdl = lm.LogisticRegression(C=1 / lambda_val)
            mdl.fit(X_train_inner, y_train_inner)

            # Calculate misclassification error on validation set
            y_val_pred = mdl.predict(X_val_inner)
            y_val_pred_train = mdl.predict(X_train_inner)
            misclass_rate_val = np.mean(y_val_pred != y_val_inner)
            misclass_rate_val_train = np.mean(y_val_pred_train != y_train_inner)
            inner_fold_errors.append(misclass_rate_val)
            inner_fold_errors_train.append(misclass_rate_val_train)

        # Average misclassification rate across inner folds
        inner_errors_test[i] = np.mean(inner_fold_errors)
        inner_errors_train[i] = np.mean(inner_fold_errors_train)

    # Select best lambda (smallest error in inner CV)
    best_lambda_index = np.argmin(inner_errors_test)
    best_lambda = lambda_interval[best_lambda_index]
    print(f"Best lambda for outer fold {outer_k + 1}: {best_lambda}")

    # Train model on entire outer training set with best lambda
    best_model = lm.LogisticRegression(C=1 / best_lambda)
    best_model.fit(X_train_outer, y_train_outer)

    # Evaluate model on outer test set
    y_test_pred = best_model.predict(X_test_outer)
    misclass_rate_test = np.mean(y_test_pred != y_test_outer)
    outer_errors.append(misclass_rate_test)
    print(f"Error of best lambda for outer fold: {misclass_rate_test}")

# Average misclassification rate across all outer folds
average_error = np.mean(outer_errors)
print("Nested CV average test misclassification rate for logistic regression:", average_error)

#Displaying the results from logistic regression with variable lambda
f = figure()
semilogx(lambda_interval, inner_errors_test * 100, label="Error_test")
semilogx(lambda_interval, inner_errors_train * 100, label="Error_train")
xlabel("Regularization (Lambda)")
ylabel("Error in % (misclassification rate, CV K={0})".format(inner_K))
legend() 
show()

# Define range of regularization parameters for K-nearest-neighbor
K_neighbor_interval = np.arange(1, 46)
outer_errors = []

for outer_k, (train_index, test_index) in enumerate(outer_cv.split(X)):
    print(f"Computing outer CV fold: {outer_k + 1}/{outer_K}...")

    # Split data into outer training and test sets
    X_train_outer, y_train_outer = X[train_index, :], y[train_index]
    X_test_outer, y_test_outer = X[test_index, :], y[test_index]

    # Standardize data based on outer training set
    mu, sigma = np.mean(X_train_outer, 0), np.std(X_train_outer, 0)
    X_train_outer = (X_train_outer - mu) / sigma
    X_test_outer = (X_test_outer - mu) / sigma

    # Inner cross-validation for hyperparameter tuning
    inner_cv = model_selection.KFold(n_splits=inner_K, shuffle=True)
    inner_errors_test = np.empty(len(K_neighbor_interval))
    inner_errors_train = np.empty(len(K_neighbor_interval))

    for i, lambda_val in enumerate(K_neighbor_interval):
        inner_fold_errors = []
        inner_fold_errors_train = []

        # Inner loop: 20-fold CV for tuning k-neighbor
        for inner_train_index, inner_val_index in inner_cv.split(X_train_outer):
            X_train_inner, y_train_inner = X_train_outer[inner_train_index], y_train_outer[inner_train_index]
            X_val_inner, y_val_inner = X_train_outer[inner_val_index], y_train_outer[inner_val_index]

            # Train model with current k-neighbor
            mdl = Knb.KNeighborsClassifier(n_neighbors=lambda_val)
            mdl.fit(X_train_inner, y_train_inner)

            # Calculate misclassification error on validation set
            y_val_pred = mdl.predict(X_val_inner)
            y_val_pred_train = mdl.predict(X_train_inner)
            misclass_rate_val = np.mean(y_val_pred != y_val_inner)
            misclass_rate_val_train = np.mean(y_val_pred_train != y_train_inner)
            inner_fold_errors.append(misclass_rate_val)
            inner_fold_errors_train.append(misclass_rate_val_train)

        # Average misclassification rate across inner folds
        inner_errors_test[i] = np.mean(inner_fold_errors)
        inner_errors_train[i] = np.mean(inner_fold_errors_train)

    # Select best lambda (smallest error in inner CV)
    best_lambda_index = np.argmin(inner_errors_test)
    best_lambda = K_neighbor_interval[best_lambda_index]
    print(f"Best K-neighbor for outer fold {outer_k + 1}: {best_lambda}")

    # Train model on entire outer training set with best lambda
    best_model = Knb.KNeighborsClassifier(n_neighbors= best_lambda)
    best_model.fit(X_train_outer, y_train_outer)

    # Evaluate model on outer test set
    y_test_pred = best_model.predict(X_test_outer)
    misclass_rate_test = np.mean(y_test_pred != y_test_outer)
    outer_errors.append(misclass_rate_test)
    print(f"Error of best K-neighbor for outer fold: {misclass_rate_test}")

# Average misclassification rate across all outer folds
average_error = np.mean(outer_errors)
print("Nested CV average test misclassification rate for K-nearst-neighbor:", average_error)

#Displaying the results from logistic regression with variable lambda
f = figure()
plot(K_neighbor_interval, inner_errors_test * 100, label="Error_test")
plot(K_neighbor_interval, inner_errors_train * 100, label="Error_train")
xlabel("Regularization (number of neighbors)")
ylabel("Error in % (misclassification rate, CV K={0})".format(inner_K))
legend() 
show()


K_neighbor_interval = np.arange(1, 46)
outer_errors = []

for outer_k, (train_index, test_index) in enumerate(outer_cv.split(X)):
    print(f"Computing outer CV fold: {outer_k + 1}/{outer_K}...")

    # Split data into outer training and test sets
    X_train_outer, y_train_outer = X[train_index, :], y[train_index]
    X_test_outer, y_test_outer = X[test_index, :], y[test_index]

    # Standardize data based on outer training set
    mu, sigma = np.mean(X_train_outer, 0), np.std(X_train_outer, 0)
    X_train_outer = (X_train_outer - mu) / sigma
    X_test_outer = (X_test_outer - mu) / sigma

    # Inner cross-validation for hyperparameter tuning
    inner_cv = model_selection.KFold(n_splits=inner_K, shuffle=True)
    inner_errors_test = np.empty(len(K_neighbor_interval))
    inner_errors_train = np.empty(len(K_neighbor_interval))

    for i, lambda_val in enumerate(K_neighbor_interval):
        inner_fold_errors = []
        inner_fold_errors_train = []

        # Inner loop: 20-fold CV for tuning k-neighbor
        for inner_train_index, inner_val_index in inner_cv.split(X_train_outer):
            X_train_inner, y_train_inner = X_train_outer[inner_train_index], y_train_outer[inner_train_index]
            X_val_inner, y_val_inner = X_train_outer[inner_val_index], y_train_outer[inner_val_index]

            # Train model with current k-neighbor
            mdl = dum.DummyClassifier()
            mdl.fit(X_train_inner, y_train_inner)

            # Calculate misclassification error on validation set
            y_val_pred = mdl.predict(X_val_inner)
            y_val_pred_train = mdl.predict(X_train_inner)
            misclass_rate_val = np.mean(y_val_pred != y_val_inner)
            misclass_rate_val_train = np.mean(y_val_pred_train != y_train_inner)
            inner_fold_errors.append(misclass_rate_val)
            inner_fold_errors_train.append(misclass_rate_val_train)

        # Average misclassification rate across inner folds
        inner_errors_test[i] = np.mean(inner_fold_errors)
        inner_errors_train[i] = np.mean(inner_fold_errors_train)

    # Select best lambda (smallest error in inner CV)
    best_lambda_index = np.argmin(inner_errors_test)
    best_lambda = K_neighbor_interval[best_lambda_index]
    print(f"Best dummy for outer fold {outer_k + 1}: {best_lambda}")

    # Train model on entire outer training set with best lambda
    best_model = dum.DummyClassifier()
    best_model.fit(X_train_outer, y_train_outer)

    # Evaluate model on outer test set
    y_test_pred = best_model.predict(X_test_outer)
    misclass_rate_test = np.mean(y_test_pred != y_test_outer)
    print(f"Error of best dummy for outer fold: {misclass_rate_test}")

    outer_errors.append(misclass_rate_test)

# Average misclassification rate across all outer folds
average_error = np.mean(outer_errors)
print("Nested CV average test misclassification rate for K-nearst-neighbor:", average_error)

#Displaying the results from logistic regression with variable lambda
f = figure()
plot(K_neighbor_interval, inner_errors_test * 100, label="Error_test")
plot(K_neighbor_interval, inner_errors_train * 100, label="Error_train")
xlabel("No regularization (its a dummy)")
ylabel("Error in % (misclassification rate, CV K={0})".format(inner_K))
legend() 
show()

#Preforming the McNiemar test
K = 10
n11, n12, n21, n22 = 0, 0, 0, 0


for i in range(K):
    CV = model_selection.KFold(n_splits=K, shuffle=True)
    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        mu, sigma = np.mean(X_train, 0), np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        KneighborModel = Knb.KNeighborsClassifier(n_neighbors= 31)
        LogRegModel = lm.LogisticRegression(C=1 / 3.2)
        KneighborModel.fit(X_train,y_train)
        LogRegModel.fit(X_train,y_train)
        y_est_LogReg = LogRegModel.predict(X_test)
        y_est_KneighborModel = KneighborModel.predict(X_test)
        for log_pred, knn_pred, true_label in zip(y_est_LogReg, y_est_KneighborModel, y_test):
            if log_pred == true_label and knn_pred == true_label:
                n11 += 1  # Both correct
            elif log_pred == true_label and knn_pred != true_label:
                n12 += 1  # A correct, B wrong
            elif log_pred != true_label and knn_pred == true_label:
                n21 += 1  # A wrong, B correct
            elif log_pred != true_label and knn_pred != true_label:
                n22 += 1  # Both wrong
        

import scipy.stats as stat
print(n12,n21)
p_value=2*stat.binom.cdf(k=min(n12,n21),n=n12+n21,p=(1/2))
  
print(p_value)
#Here is the final model selected
#I wish to see the weights

mdl = lm.LogisticRegression(C=1 / 3.2)
mu, sigma = np.mean(X, 0), np.std(X, 0)
X_final = (X - mu) / sigma
mdl.fit(X_final, y)
print()
print("Best model coefficients:", mdl.coef_)
