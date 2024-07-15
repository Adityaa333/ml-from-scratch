import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = 0.001
n_iters = 1000
degree = 2 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fit_transform(x):
    # Z-score = current - mean / std deviation
    z = (x - np.mean(x)) / np.std(x)
    return z
#standardizing increases accuracy from 0.8947 to 0.9035  

def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    output_features = [np.ones(n_samples)]
    
    for d in range(1, degree + 1):
        for item in range(n_features):
            output_features.append(X[:, item] ** d)
    
    return np.column_stack(output_features)
#using poly features increases accuracy from 0.9035 to  0.9210 

def fit(X, y, lr, n_iters):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iters):
        linear_pred = np.dot(X, weights) + bias
        predictions = sigmoid(linear_pred)

        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)

        weights -= lr * dw
        bias -= lr * db
    
    return weights, bias

def predict(X, weights, bias):
    linear_pred = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_pred)
    class_pred = [0 if i <= 0.5 else 1 for i in y_pred]
    return class_pred

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

X_train_std  , X_test_std = fit_transform(X_train) , fit_transform(X_test)
X_train_std_poly , X_test_std_poly = polynomial_features(X_train_std , degree) , polynomial_features(X_test_std , degree)

# weights, bias = fit(X_train, y_train, lr, n_iters)
# y_pred = predict(X_test, weights, bias)

# weights, bias = fit(X_train_std , y_train, lr, n_iters)
# y_pred = predict(X_test_std, weights, bias)

weights, bias = fit(X_train_std_poly , y_train, lr, n_iters)
y_pred = predict(X_test_std_poly, weights, bias)

print("Accuracy :" , accuracy(y_pred , y_test))

cm = confusion_matrix(y_test , y_pred)
tn, fp, fn, tp = confusion_matrix(y_test , y_pred).ravel()
print("Recall = " , tp/(tp + fn))
print("Precision = " , tp/(tp + fp))
sns.heatmap(cm , annot=True )
