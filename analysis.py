# Import libraries for analysis
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create dataframe from training data set
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("titanic_test")

# Replace missing ages with median age from the training data
median_age = titanic["Age"].median()
titanic_test["Age"] = titanic_test["Age"].fillna(median_age)

# Convert male to 0 and female to 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Replace missing values with most common port which is S
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# Assign numbers to S, C and Q
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Replace missing fare value with median from test set
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Use logistic regression to output values between 0 and 1
logreg_algorithm = LogisticRegression(random_state=1)

# Use all training data to train algorithm
logreg_algorithm.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set
predictions = logreg_algorithm.predict(titanic_test[predictors])

# Create a dataframe with requested columns for Kaggle submission
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived":predictions})

# Output results to csv file
submission.to_csv("titanic.csv", index=False)