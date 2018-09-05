#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:33:14 2018

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

# we need to use feature scaling
# In LinearRegression class feature scaling is included
# but here we are using a less common class and we need to do it explicitly
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)'''

# we are not splitting the data to train and test set
# reason being we have very few amount of data
# also so that we make an accurate prediction

# fit polynomial linear regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualisation: Decision Tree for higher resolution and smoother curve
# can be used with any linear model as well
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)

y_pred = regressor.predict(X_grid)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, y_pred, color = 'blue')
plt.title('Salary Truth(Decision Tree)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Prediction: svr
regressor.predict(6.5)
