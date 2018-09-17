# Data Preprocessing

# Importing the libraries
# numpy is used to perform mathematical operations
# Matplotlib is used for plotting- to plot charts
# pandas is used to import datasets and for creating dataframes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# In this case, the independent variables are country, age, salary, purchased.
# and purchased is the dependent variable. So we will take the independent variable in X. 
#[:,:-1] - this first ':' means that include all rows, the second ':-1' means include all columns except the last column which in this case is salary
# This is known as matrix of features
X = dataset.iloc[:, :-1].values
# Now make matrix of features for the dependent variable which is purchased.
#'3' in this case picks up the third column because index starts from zero
y = dataset.iloc[:, 3].values

# Taking care of missing data
# This dataset has missing values which are mentioned as 'nan'. This disrupts the model.
# We need to replace the missing values- one approach would be to replace the 'nan' with the mean of that column
# To achieve this, import 'Imputer' from sklearn.preprocessing
from sklearn.preprocessing import Imputer
# This will define the strategy and save it in a variable imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Apply the imputer on the second and third column. [1:3] means the columns with index 1 and 2
imputer = imputer.fit(X[:, 1:3])
# Transform the matrix
X[:, 1:3] = imputer.transform(X[:, 1:3])
# You can print the matrix of features X to see that 'nan' has now been replaced
print(X)

# Encoding Categorical Data- Categorical data has to be converted into numerical form
# because machines understand only numbers
# In our case, country is categorical data

# Let's encode the independent variable 'X'. It contains the categorical variable "country"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Encode the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
