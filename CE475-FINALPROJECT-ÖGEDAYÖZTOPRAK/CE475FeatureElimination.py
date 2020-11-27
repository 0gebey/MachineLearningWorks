# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:08:44 2020

@author: ogeda
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

data = pd.read_csv('CE475FINALPROJECTDATA.csv', index_col=0, nrows=100)
columns = ['x1', 'x2', 'x3','x4','x5','x6']
X = data[columns]
y = data["Y"]

regression = LinearRegression()


def feature_selection(X, y):
	rfe = RFE(regression,n_features_to_select=4)
	fit = rfe.fit(X, y)
	print("Feature Ranking: ", fit.ranking_)
	return fit.transform(X)
feature_selection(X, y)

print('These are the columns of our data in the sorted version according to Recursive Feature Elimination.')
print('So I have decided to work with first four data which are; x1,x2,x3,x5 for the further applications.')