#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:15:25 2018

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# we are not splitting the data to train and test set
# reason being we have very few amount of data
# also so that we make an accurate prediction

# fit polynomial linear regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Visualisation: svr
y_pred = regressor.predict(X)
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Salary Truth(SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Prediction: svr
sc_y.inverse_transform(regressor.predict(sc_X.transform(6.5)))

