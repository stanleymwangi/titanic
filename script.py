# Import libraries for analysis
import pandas as pd
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

# Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialise algorithm class
linreg_algorithm = LinearRegression()

# Make cross-validation folds for titanic data set
kf = KFold(n_splits=3, random_state=1)  # setting random_state ensures splits stay the same on each run

predictions = []
for train, test in kf.split(titanic): # kf.split produces row indices for training and testing
    count +=1

    # Predictors for training the algorithm from the training fold
    train_predictors = (titanic[predictors].iloc[train,:])

    # Target for training the algorithm from the training fold
    train_target = titanic["Survived"].iloc[train]

    # Train algorithm using predictors and target
    linreg_algorithm.fit(train_predictors, train_target)

    # Apply trained algorithm to test fold to make predictions
    test_predictions = linreg_algorithm.fit(titanic[predictors].iloc[test,:])

    # Add resulting predictions
    predictions.append(test_predictions)