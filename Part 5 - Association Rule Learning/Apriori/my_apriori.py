#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:17:03 2018

@author: ishaMac
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# apriori function expects list of transactions
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
# minsup is calculated for items purchased 3 times a day
# sup = 3*7/7500
# minconf should not be too high, otherwise we won't get the obvious rules

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
res = []
for i in range(0, len(results)):
    res.append("Rule:\t" + str(results[i][0]) + "\nSupport:\t" + str(results[i][1]))