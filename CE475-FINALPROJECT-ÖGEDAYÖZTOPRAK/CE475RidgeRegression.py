# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:58:47 2020

@author: ogeda
"""


from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import linear_model

data = pd.read_csv('CE475FINALPROJECTDATA.csv', encoding='latin1', nrows=100)
columns = ['x1', 'x2', 'x3', 'x5']
X= data[columns]
y= data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

ridgeReg = linear_model.Ridge(alpha=10)
ridgeReg.fit(X_train,y_train)
y_pred= ridgeReg.predict(X_test)
CRS = cross_val_score(ridgeReg, X, y, cv=10)

print('-- Ridge Regression Results --')
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error of the Data:', MAE)
MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error of the Data:', MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error of the Data:',RMSE)
CVA = format(np.mean(CRS))
print('CVA Score of the Data: ', CVA )
ACC = r2_score(y_test, y_pred)
print('Accuracy of the Data', ACC )
