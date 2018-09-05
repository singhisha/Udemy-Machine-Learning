#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:11:01 2018

@author: ishaMac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# we are not splitting the data to train and test set
# reason being we have very few amount of data
# also so that we make an accurate prediction

# fit polynomial linear regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Prediction: svr
regressor.predict(6.5)

# Visualisation: svr for higher resolution and smoother curve
# can be used with any linear model as well
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)

y_pred = regressor.predict(X_grid)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, y_pred, color = 'blue')
plt.title('Salary Truth(Random Forest Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


