# Import libraries for analysis
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create dataframe from training data set
titanic = pd.read_csv("train.csv")

# Age column has missing values as count is not 891
# Replace missing ages with median age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Convert male to 0 and female to 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Replace missing values with most common port which is S
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# Assign numbers to S, C and Q
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Use logistic regression to output values between 0 and 1
logreg_algorithm = LogisticRegression(random_state=1)

# Use cross_val_score for cross-validation and evaluation
scores = cross_val_score(logreg_algorithm, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores produced by the folds
print(scores.mean())
