#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:32:30 2018

@author: jasleenarora
"""

# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Fitting the decision tree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
# Fit the regressor object to the dataset
regressor.fit(X, y)

# Predicting the new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()

# It has no predictions to plot in the intervals, so it makes a line between point
# this is non-linear and non-continous regression model
# the best way to visualise the results in higher dimensions
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision tree regression')
plt.show()
# this shows the non-continuity
# Now we can clearly see the intervals

# Let's check the salary of employee with 6.5 now
y_pred = regressor.predict(6.5)