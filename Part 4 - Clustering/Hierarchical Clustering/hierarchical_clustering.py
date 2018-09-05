#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:39:50 2018

@author: ishaMac
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as hc
dendogram = hc.dendrogram(hc.linkage(X, method = 'ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Fit hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
ag = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')
y = ag.fit_predict(X)

# Visualisation
plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 60, c = 'red', label = 'Careful')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 60, c = 'blue', label = 'Average')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s = 60, c = 'green', label = 'Target')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s = 60, c = 'cyan', label = 'Careless')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s = 60, c = 'magenta', label = 'Sensible')
plt.title("Client Clusters")
plt.xlabel("Annual Income($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()