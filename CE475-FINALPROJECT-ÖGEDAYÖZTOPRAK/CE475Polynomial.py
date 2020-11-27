# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:01:57 2020

@author: ogeda
"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

data = pd.read_csv('CE475FINALPROJECTDATA.csv', encoding='latin1', nrows=100)
columns = ['x1', 'x2', 'x3', 'x5']
X= data[columns]
y= data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

degrees = np.arange(1, 10)
min_rmse, min_deg = 1e10, 0
rmses = []

for deg in degrees:

    # Train features
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly_train = poly_features.fit_transform(X_train)

    # Linear regression
    polynomial_reg = LinearRegression()
    polynomial_reg.fit(x_poly_train, y_train)

    # Compare with test data
    x_poly_test = poly_features.fit_transform(X_test)
    poly_predict = polynomial_reg.predict(x_poly_test)
    poly_mse = mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)

    # Cross-validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg

# Plot and present results
print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')

# Polynomial Regression Starts With Degree 2

polynomial_reg = PolynomialFeatures(degree=2)
X_polinomial = polynomial_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg_with_pure_values = lin_reg.fit(X_polinomial, y)

CRS = cross_val_score(lin_reg_with_pure_values, X, y, cv=10)
y_pred = lin_reg.predict(polynomial_reg.fit_transform(X_test))
# Results
print('-- Polynomial Regression Results --')
print('Best degree was {} with RMSE {}'.format(min_deg, min_rmse))
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error of the Data:', MAE )
MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error of the Data:', MSE )
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error of the Data:', RMSE )
CVA = format(np.mean(CRS))
print('CVA Score of the Data:' , CVA )
ACC = r2_score(y_test, y_pred)
print('Accuracy of the Process', ACC )
  
    
    


