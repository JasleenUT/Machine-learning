#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:29:43 2018

@author: jasleenarora
"""
#   MULTIPLE LINEAR REGRESSION

# This dataset contains many predictors
# This model will allow them that which areas should be invested in to maximize profit

# here the dependent variable is dependent on multiple factors
# i.e. it has many independent coefficients

# Dummy variables - to convert categorical predictors to numeric ones
# Our dataset has to be expanded such that the number of distinct values in the categorical column
# will be equal to the number of new columns that will be created
# All dummy variable columns need not be a part of your regression model
 
# Preprocess the data

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Add dummy variables for encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:] #removed the first column from X
# regression library does this but it is better to do it manually

#Split data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit multiple Linear regression model to training set
from sklearn.linear_model import LinearRegression
#create an object of linear regression class
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test Set Results
# we cannot create a plot because it is difficult to see 5 dimensions
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination

# Preparation for backward elimination
import statsmodels.formula.api as sm
# the statsmodels library does not take into account the constant
# so we need to add a column of 1 in matrix of features which will correspond to x0=1
X = np.append(arr = np.ones((50,1)).astype(int), values =X, axis = 1)
# the ones will create a matrix having only 1 as its values, axis = 1 is to add a column and axis = 0 is used to add a line
 
# Starting Backward elimination
# create new optimal matrix of features- having variables that are statistically significant on the profit
# add all the columns in the first step
X_opt = X[:,[0,1,2,3,4,5]]

# Select a significance level i.e. alpha, I have chosen 0.05
 



















