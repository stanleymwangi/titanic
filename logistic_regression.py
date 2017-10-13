# Import libraries for analysis
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create dataframe from training data set
titanic = pd.read_csv("clean_train.csv")

# Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Use logistic regression to output values between 0 and 1
logreg_algorithm = LogisticRegression(random_state=1)

# Use cross_val_score for cross-validation and evaluation
scores = cross_val_score(logreg_algorithm, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores produced by the folds
print(scores.mean())
