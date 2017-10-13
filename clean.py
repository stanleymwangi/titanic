# Import libraries for analysis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold # helper for cross validation

# Create dataframe from training data set
titanic = pd.read_csv("train.csv")

# Display passenger data for first five rows
print(titanic.head(5))

# Show summary stats for numeric columns
print(titanic.describe())

# Age column has missing values as count is not 891
# Replace missing ages with median age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Gender is currently non-numeric but could be useful in predictions
print(titanic["Sex"].unique())

# Convert male to 0 and female to 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Show current unique values in embarked column
print(titanic["Embarked"].unique())

# Replace missing values with most common port which is S
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# Assign numbers to S, C and Q
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Output cleaned version of train.csv  results to csv file
titanic.to_csv("clean_train.csv", index=False)