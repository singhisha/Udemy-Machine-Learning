#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:49:31 2018

@author: ishaMac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# we are not splitting the data to train and test set
# reason being we have very few amount of data
# also so that we make an accurate prediction

# fit polynomial linear regression
from sklearn.linear_model import LinearRegression
reg_linear = LinearRegression()
reg_linear.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
reg_poly = PolynomialFeatures(degree = 4)
X_poly = reg_poly.fit_transform(X)
reg_linear2 = LinearRegression()
reg_linear2.fit(X_poly, y)

# Visualisation: linear vs polynomial
y_pred = reg_linear.predict(X)
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Salary Truth(Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

y_pred2 = reg_linear2.predict(X_poly)
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred2, color = 'blue')
plt.title('Salary Truth(Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Prediction: Linear vs Polynomial
reg_linear.predict(6.5)

reg_linear2.predict(reg_poly.fit_transform(6.5))