import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Generate dataset
X, y = datasets.make_regression(n_samples=500, n_features=1, noise=35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Ridge regression with hyperparameter tuning
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train_poly, y_train)
best_ridge_reg = grid_search.best_estimator_
y_pred_ridge = best_ridge_reg.predict(X_test_poly)

# Print the best alpha parameter
print(f"Best alpha for Ridge regression: {grid_search.best_params_['alpha']}")

# Linear regression with gradient descent
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = predict(X, weights, bias)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, weights, bias, learning_rate, n_iterations):
    m = len(y)
    cost_history = np.zeros(n_iterations)

    for i in range(n_iterations):
        predictions = predict(X, weights, bias)
        errors = predictions - y

        weights_gradient = (1/m) * np.dot(X.T, errors)
        bias_gradient = (1/m) * np.sum(errors)

        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient

        cost_history[i] = compute_cost(X, y, weights, bias)
    
    return weights, bias, cost_history

def train(X, y, learning_rate, n_iterations):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0

    weights, bias, cost_history = gradient_descent(X, y, weights, bias, learning_rate, n_iterations)
    return weights, bias, cost_history

learning_rate = 0.1
n_iterations = 1000
weights, bias, cost_history = train(X_train_scaled, y_train, learning_rate, n_iterations)
y_pred_linear = predict(X_test_scaled, weights, bias)

# Plot cost reduction over time for linear regression
plt.plot(range(n_iterations), cost_history)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Cost reduction over time for Linear Regression")
plt.show()

# Plot results comparison
plt.figure(figsize=(14, 6))

# Actual data
plt.scatter(X_test, y_test, color='black', label='Actual data')

# Linear regression results
plt.plot(X_test, y_pred_linear, color='red', label='Linear Regression')

# Ridge regression results
plt.plot(X_test, y_pred_ridge, color='blue', label='Ridge Regression')

plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression vs Ridge Regression")
plt.legend()
plt.show()
