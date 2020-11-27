# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:22:38 2020

@author: ogeda
"""
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

data = pd.read_csv('CE475FINALPROJECTDATA.csv', encoding='latin1', nrows=100)
columns = ['x1', 'x2', 'x3', 'x5']
X= data[columns]
y= data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

decReg = DecisionTreeRegressor(random_state=0)
decReg.fit(X_train, y_train)
y_pred = decReg.predict(X_test)
CRS = cross_val_score(decReg, X, y, cv=10)
print("--Decision Tree Results--")
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error of the Data:', MAE)
MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error of the Data:', MSE )
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error of the Data:', RMSE )
CVA = format(np.mean(CRS))
print('CVA score of the Data: ' , CVA )
ACC = r2_score(y_test, y_pred)
print('Accuracy of the Data:', ACC )
