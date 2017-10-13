# Import libraries for analysis
import pandas as pd

# Create dataframe from training data set
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")

# Display passenger data for first five rows
print(titanic.head(5))

# Show summary stats for numeric columns
print(titanic.describe())

# Age column has missing values as count is not 891
# Replace missing ages with median age
median_age = titanic["Age"].median()

titanic["Age"] = titanic["Age"].fillna(median_age)
# Replace missing ages with median age from the training data
titanic_test["Age"] = titanic_test["Age"].fillna(median_age)

# Gender is currently non-numeric but could be useful in predictions
print(titanic["Sex"].unique())

# Convert male to 0 and female to 1 in both training and test
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Show current unique values in embarked column
print(titanic["Embarked"].unique())

# Replace missing values with most common port which is S
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# Assign numbers to S, C and Q
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Replace missing fare value with median from test set
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# Output cleaned version of train.csv  results to csv file
titanic.to_csv("clean_train.csv", index=False)
titanic_test.to_csv("clean_test.csv", index=False)