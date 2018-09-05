#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:52:34 2018

@author: ishaMac
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# use the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#fit_predict will give the cluster number for each point
y = kmeans.fit_predict(X)

# Visualisation
plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 60, c = 'red', label = 'Careful')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 60, c = 'blue', label = 'Average')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s = 60, c = 'green', label = 'Target')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s = 60, c = 'cyan', label = 'Careless')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s = 60, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroid')
plt.title("Client Clusters")
plt.xlabel("Annual Income($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()