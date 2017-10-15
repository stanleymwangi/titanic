# Import libraries for analysis
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

# Create dataframes from clean training data and test data
titanic = pd.read_csv("clean_train.csv")
titanic_test = pd.read_csv("clean_test.csv")

# # Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Use logistic regression to output values between 0 and 1
rf_classifier = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
