# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:19:14 2018

@author: ak901t
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear regression to the dataset.
from sklearn.linear_model import LinearRegression
linearRegressor1 = LinearRegression()
linearRegressor1.fit(X,y);

# Fitting polynomial regression to the dataset.
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree=4)
X_poly = polynomialRegressor.fit_transform(X)
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_poly , y)

# Visualising the linear regression results
plt.scatter(X, y, c='Red')
plt.plot(X, linearRegressor1.predict(X), c='Blue')
plt.title('Truth or Bluff (Linear Regression Results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid) , 1)
plt.scatter(X, y, c='Red')
plt.plot(X_grid, linearRegressor2.predict(polynomialRegressor.fit_transform(X_grid)), c='Blue')
plt.title('Truth or Bluff (Polynomial Regression Results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression.
linearRegressor1.predict(6.5)

# Predicting a new result with Polynomial Regression.
linearRegressor2.predict(polynomialRegressor.fit_transform(6.5))