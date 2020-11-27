# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:41:22 2020

@author: ogeda
"""

from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('CE475FINALPROJECTDATA.csv', encoding='latin1', nrows=100)
columns = ['x1', 'x2', 'x3', 'x5']
data2 = pd.read_csv('CE475FINALPROJECTDATA.csv', index_col=0, skiprows=range(1, 101), nrows=20)
X= data[columns]
y= data['Y']
y_WillPredicted = data2[columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

RF_reg = RandomForestRegressor(random_state=0, max_features='auto')
RF_reg = RF_reg.fit(X_train, y_train)
CRS = cross_val_score(RF_reg, X, y, cv=10)
y_pred = RF_reg.predict(X_test)

print('-- Random Forest Regression Results --')
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error of the Data::', MAE )
MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error of the Data:', MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared of the Data:', RMSE )
CVA = format(np.mean(CRS))
print('CVA score of the Data:' , CVA )
ACC = r2_score(y_test, y_pred)
print('Accuracy of the Data:', ACC )

newyvalues_pred = RF_reg.predict(y_WillPredicted)

print("Random Forest = ", newyvalues_pred[0:20])