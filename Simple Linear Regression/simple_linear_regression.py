# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# FITTING THE LINEAR REGRESSION MODEL ON TRAINING SET

# Import library
from sklearn.linear_model import LinearRegression
# create a regressor
regressor = LinearRegression()
# fit the object to training set
regressor.fit(X_train, y_train)
# A machine has been made 'regressor' which can now predict salary according to the experience

#PREDICTING THE TEST RESULT

# a vector of predictions of dependent variable
# predict method makes predictions
# input the test set
y_pred = regressor.predict(X_test)

# Now y_test is the actual salary of the people in the company
# and y_pred is the salary predicted by our model

# Because simple linear regression is a straight line,
# it oversetimates in some cases and underestimates in some

# Visualising the training set results

# Plot the graphs and visulaize the results
#plot the real observation point
# x-axis will have experience and y- salaries
# observation points in red and regression line in blue
plt.scatter(X_train, y_train, color = 'red')
# make the regression line
# y co-ordinates will be the predictions of X_train
plt.plot(X_train, regressor.predict(X_train), color='blue')
# Add a heading to the plot
plt.title('Salary vs Experience (Training Set)')
# Add a label to the x-axis
plt.xlabel('Years of experience')
# Label on y-axis
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
# Add a label to the x-axis
plt.xlabel('Years of experience')
# Label on y-axis
plt.ylabel('Salary')
plt.show()



