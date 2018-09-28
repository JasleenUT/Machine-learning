#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:45:25 2018

@author: jasleenarora
"""

# Polynomial Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #to make X as matrix of features
y = dataset.iloc[:,2].values
# Make sure that X is a matrix and y is  a vector

# the HR has to hire a new employee and the salary has to be decided
# the employee is asking 160k and has 20 years of experience

# The position has already been encoded in the level.
# thus only level is required in matrix of features

#Goal is to predict the salary of 6.5 level which has been identified by the employee
# we don't need to split the dataset into training and test set because we have in hand our test and our dataset size is very small

# Fit a linear regression as a reference base
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Transform matrix of features X into a new matrix of features 
# having x1^2,x1^3 and so on 
poly_reg = PolynomialFeatures(degree = 2) #start with degree 2
X_poly = poly_reg.fit_transform(X)

# Now open and see X_poly. The third column will be the square of the second column
# poly_reg has created a column of 1 to include the constant
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Polynomial regression model has been created

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red') #real observation points
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red') #real observation points
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# If you see the salary for 6.5, the employee seems  honest as it shows 200000

# Now let's make the model 3 degree
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Transform matrix of features X into a new matrix of features 
# having x1^2,x1^3 and so on 
poly_reg = PolynomialFeatures(degree = 3) #start with degree 2
X_poly = poly_reg.fit_transform(X)

# Now open and see X_poly. The third column will be the square of the second column
# poly_reg has created a column of 1 to include the constant
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Visualizing the curve now, the prediction is much better to the real values
# We can say that the employee for saying the truth

# lets make a degree 4
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# The prediction in this case is really accurate
# Instead of having straight lines, we can have incremental curves
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red') #real observation points
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))



