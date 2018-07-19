#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:12:48 2018

@author: ishaMac
"""
# this template can be used for any machine learning model by making the necessary changes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Data.csv")
X = data.iloc[:,:3].values
y = data.iloc[:,-1].values

# replace missing data by the mean of the entire column
from sklearn.preprocessing import Imputer
impute = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
impute = impute.fit(X[:,1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])

# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)