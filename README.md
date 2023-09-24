Predicting MPG using Linear Regression Algorithm
This project demonstrates the use of a linear regression algorithm to predict the miles per gallon (MPG) of a car based on various features. Linear regression is a simple yet effective machine learning technique for regression tasks like this one.

Import Libraries
Before we get started, make sure you have the necessary libraries installed. You can install them using pip if you haven't already.

python
Copy code
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
Load and Review Data
Let's start by loading and reviewing the dataset.

python
Copy code
# Load the dataset
cData = pd.read_csv('auto-mpg.csv')

# Display the first 10 rows of the dataset
cData.head(10)
The dataset contains information about various car attributes, including MPG, cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name.

Data Preprocessing
Create Dummy Variables
We need to create dummy variables for the 'origin' column, as it contains categorical data. This will allow us to use it as a feature in our linear regression model.

python
Copy code
# Replace 'origin' with dummy variables
cData['origin'] = cData['origin'].replace({1: 'america', 2: 'europe', 3: 'japan'})

# Display the updated dataset
cData.head()
Handling Missing Values
We should check for missing values in the dataset and replace them with appropriate values. In this case, we will replace missing values in the 'horsepower' column with the median value.

python
Copy code
# Check for missing values
cData.isnull().sum()

# Replace missing values in 'horsepower' with the median
median_horsepower = cData['horsepower'].median()
cData['horsepower'] = cData['horsepower'].replace('?', median_horsepower)
cData['horsepower'] = cData['horsepower'].astype('float64')

# Verify that there are no more missing values
cData.isnull().sum()
Creating Dummy Variables
Next, we'll create dummy variables for the 'origin' column.

python
Copy code
# Create dummy variables for 'origin'
cData = pd.get_dummies(cData, columns=['origin'])

# Display the updated dataset with dummy variables
cData.head()
Data Visualization
Let's visualize the data to gain insights before building the model.

python
Copy code
# Bivariate plots
cData_attribute = cData.iloc[:, 0:7]
sns.pairplot(cData_attribute, diag_kind='kde')
Splitting the Data
Before training the model, we need to split the dataset into training and testing sets.

python
Copy code
# Define the dependent variable (MPG) and independent variables (features)
y = cData['mpg']
X = cData.drop('mpg', axis=1)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
Training the Model
Now, we'll train a linear regression model using the training data.

python
Copy code
# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients for each feature
for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, coefficients[idx]))

# Calculate the R-squared score for the training and testing data
train_score = model.score(X_train, y_train) * 100
test_score = model.score(X_test, y_test) * 100

# Make predictions on the test data
predictions = model.predict(X_test)
Model Evaluation
We have trained our linear regression model. Here are some key results:

Coefficients for each feature indicate their influence on MPG.
R-squared scores provide a measure of model performance on both training and testing data.
Predictions have been made on the test data.
Feel free to use this linear regression model to predict MPG for new data or further analyze the results.
