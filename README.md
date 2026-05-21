# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2. Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3. Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4. Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.
   

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Poornima J 
RegisterNumber:  212225040303

import numpy as np

# Input Features
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target
y = np.array([50, 45, 70, 80, 95], dtype=float)

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Hyperparameters
lr = 0.0001
epochs = 1000

# SGD Training
for epoch in range(epochs):

    for i in range(len(X)):

        # Prediction
        y_pred = np.dot(X[i], w) + b

        # Error
        error = y_pred - y[i]

        # Update weights and bias
        w = w - lr * error * X[i]
        b = b - lr * error

# Final weights
print("Weights:", w)
print("Bias:", b)

# Prediction
predictions = np.dot(X, w) + b

print("\nPredictions:")
print(predictions)
*/
```

## Output:

Weights: [ 0.94291031 -0.15276151  1.15828785]
Bias: 0.10111469708778799

Predictions:
[47.68040697 40.09566899 72.14727978 86.37978644 98.32087045]


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
