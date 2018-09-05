#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:23:04 2018

@author: ishaMac
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the dummy variable trap, btw we don't need to do it manually
# libraries are already taking care of it
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#prediction
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
# append X to the column of ones because we want ones in the beginning of X
# we use all the rows and we don't split into test and train 
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
sig = 0.05
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()
# removing bases on the p-value
# remove that parmeter if p-value>sig
X_opt = X[:, [0,1,3,4,5]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3,4,5]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3,5]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()