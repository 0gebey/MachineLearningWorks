# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:27:46 2020

@author: ogeda
"""

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np

data = pd.read_csv('CE475FINALPROJECTDATA.csv', encoding='latin1', nrows=100)
columns = ['x1', 'x2', 'x3', 'x5']
X= data[columns]
y= data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear Regression

regr = LinearRegression() 
regr.fit(X_train, y_train) 
CRS = cross_val_score(regr, X_train, y_train, cv=10)
y_pred = regr.predict(X_test)

#Results

print('--Linear Regression Results --')
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error of the Data:' , MAE)
MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error of the Data:' , MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error of the Data', RMSE)
CVA = format(np.mean(CRS))
print('CVA Score of the Data:', CVA)
ACC = r2_score(y_test, y_pred)  
print('Accuracy of the Process:' , ACC)
