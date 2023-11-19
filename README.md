## Predicting MPG using Linear Regression Algorithm

### Introduction

The goal of this analysis is to predict the Miles Per Gallon (MPG) of cars using the Linear Regression algorithm. The dataset used for this analysis contains various attributes such as cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name. The steps involved in the analysis include data loading and review, handling missing values, feature engineering, creating dummy variables, data visualization, splitting the dataset, training the model, and evaluating its performance.

### Steps

#### 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

#### 2. Loading and Reviewing Data

```python
cData = pd.read_csv('auto-mpg.csv')
cData.head(10)
```

#### 3. Creating Dummy Variables

```python
cData['origin'] = cData['origin'].replace({1:'america',2:'europe',3:'japan'})
cData = pd.get_dummies(cData, ['origin'])
```

#### 4. Handling Missing Values

```python
cData = cData.replace({'?':np.nan})
cData['horsepower'] = cData['horsepower'].replace(np.nan, cData['horsepower'].median())
```

#### 5. Feature Engineering

```python
cData = cData.drop('car name', axis=1)
```

#### 6. Data Visualization

```python
cData_attribute = cData.iloc[:, 0:7]
sns.pairplot(cData_attribute, diag_kind='kde')
```

#### 7. Splitting the Model

```python
y = cData['mpg']
x = cData.drop('mpg', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
```

#### 8. Training the Model

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

#### 9. Model Evaluation

```python
model.score(x_test, y_test) * 100
```

### Source

The dataset used in this analysis is sourced from the 'auto-mpg' dataset, which is commonly used in machine learning and data science tutorials. The dataset contains information about various car models and their specifications.

### Conclusion

The Linear Regression model achieved a certain accuracy percentage in predicting the MPG of cars based on the given features. This analysis demonstrated the typical steps involved in a data analysis project, including data preprocessing, feature engineering, model training, and evaluation. The predictive model can be further fine-tuned or replaced with more complex models for improved accuracy, depending on the specific requirements of the application.
