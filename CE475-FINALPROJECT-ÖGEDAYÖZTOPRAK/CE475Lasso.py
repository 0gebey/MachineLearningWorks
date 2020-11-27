# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:24:58 2020

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


lasso_model = linear_model.Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
pred = lasso_model.predict(X_test)
CRS = cross_val_score(lasso_model, X, y, cv=5)


print('-- Lasso Regression Results --')
MAE = metrics.mean_absolute_error(y_test, pred)
print('Mean Absolute Error of the Data:', MAE)
MSE = metrics.mean_squared_error(y_test, pred)
print('Mean Squared Error of the Data:', MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error of the Data:',RMSE)
CVA = format(np.mean(CRS))
print('CVA Score of the Data: ', CVA )
ACC = r2_score(y_test, pred)
print('Accuracy of the Data', ACC )




