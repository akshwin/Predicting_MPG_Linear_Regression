# Predicting MPG Using Linear Regression Algorithm
# Introduction
In this project, we will explore how to predict a car's miles per gallon (MPG) using a Linear Regression algorithm. The dataset used for this project contains information about various car attributes like the number of cylinders, displacement, horsepower, weight, acceleration, model year, and origin. We aim to build a model that can accurately predict a car's MPG based on these attributes.

# Summary of Steps
Here's an overview of the steps we've followed in this project:

1. Import Libraries
We start by importing the necessary Python libraries for data analysis and modeling. These libraries include NumPy, Pandas, Seaborn, Matplotlib, and scikit-learn's LinearRegression model. We also enable inline plotting for Matplotlib and suppress warnings.

2. Load and Review Data
We load the dataset from a CSV file named 'auto-mpg.csv' and display the first 10 rows to get a glimpse of the data. The dataset contains information about 398 cars and their attributes.

3. Create Dummy Variables
We convert the 'origin' column, which represents the car's country of origin, into dummy variables using one-hot encoding. This helps us include this categorical variable in our regression model.

4. Feature Engineering
We drop the 'car name' column as it's not used for predicting MPG. Additionally, we handle missing values in the 'horsepower' column by replacing them with the median value. We also convert the 'horsepower' column to a numeric data type.

5. Creating Dummy Variables
We create dummy variables for the 'origin' column using one-hot encoding, which converts the categorical variable into numerical format.

6. Visualization
We generate bivariate plots to visualize the relationships between different attributes and the target variable (MPG). These plots help us understand the data's distribution and correlations.

7. Splitting the Model
We split the dataset into training and testing sets, with 70% of the data used for training the model and 30% for testing its performance.

8. Training the Model
We use scikit-learn's LinearRegression model to train our prediction model on the training data. We calculate the coefficients and intercept to understand the linear relationship between the independent variables and MPG.

9. Model Evaluation
We evaluate the model's performance by calculating the R-squared score on both the training and testing data. The R-squared score measures the proportion of the variance in the dependent variable (MPG) that is predictable from the independent variables.

# Sources
The dataset used in this project, 'auto-mpg.csv,' is a common dataset available for educational purposes and can be found in various online resources and data science platforms.

# Conclusion
In this project, we successfully built a Linear Regression model to predict a car's MPG based on its attributes. The model achieved an R-squared score of approximately 81.41% on the training data and 84.33% on the testing data, indicating that it provides reasonably accurate predictions for MPG. This predictive model can be useful for various applications, such as assessing a car's fuel efficiency or making informed decisions about vehicle design and manufacturing.
