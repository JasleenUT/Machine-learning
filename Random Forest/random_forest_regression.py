#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:14:48 2018

@author: jasleenarora
"""

# Random Forest Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Creating regressor
from sklearn.ensemble import RandomForestRegressor
# default is random forest of 10 trees = n_estimators
regressor = RandomForestRegressor(n_estimators=10, random_state = 0)
regressor.fit(X,y)

# Visualising the regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision tree regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# If we increase the trees, it does not mean that there will be more steps,
# they will converger towards the average

# Now let's predict the salary
y_pred = regressor.predict(6.5)

# you can increase the trees to see if the prediction gets better
# Rebuilding the model with 100 trees
regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor.fit(X,y)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision tree regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# see the prediction now
y_pred = regressor.predict(6.5)
# we can see that the prediction now is 158k which is much closer to the value he quoted

# Let's increase the trees to 300
regressor = RandomForestRegressor(n_estimators=300, random_state = 0)
regressor.fit(X,y)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision tree regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred = regressor.predict(6.5)
# we can now see that the predicted value now is 160k











