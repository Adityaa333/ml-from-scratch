import numpy as np 
import pandas as pd 
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_samples=500, n_features=1, noise=35, random_state=42)
X_train , X_test,  y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state=43)

#parameters 
learning_rate = 0.1

#MSE with lr=0.001 = 1738.0710041393468
#MSE with lr=0.01 = 1163.6676855180447
#MSE with lr=0.1 = 1163.6647636746434

weights = None
bias = None
n_iterations = 1000  

#hypothesis func
def predict(X , weights , bias):
    return np.dot(X , weights) + bias

#MSE
def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = predict(X, weights, bias)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

#training model
def train(X, y, learning_rate, n_iterations):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0

    weights, bias, cost_history = gradient_descent(X, y, weights, bias, learning_rate, n_iterations)
    return weights, bias, cost_history

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

weights, bias, cost_history = train(X_train, y_train, learning_rate, n_iterations)
y_pred = predict(X_test, weights, bias)
print(f"Weights: {weights}")
print(f"Bias: {bias}")

plt.plot(range(n_iterations), cost_history)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Cost reduction over time")
plt.show()

# Plot results
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Fitted line')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
