# Import libraries for analysis
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Create dataframes from clean training data and test data
titanic = pd.read_csv("clean_train.csv")
titanic_test = pd.read_csv("clean_test.csv")

# # Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Use logistic regression to output values between 0 and 1
logreg_algorithm = LogisticRegression(random_state=1)
#
# # Use all training data to train algorithm
logreg_algorithm.fit(titanic[predictors], titanic["Survived"])
#
# # Make predictions using the test set
predictions = logreg_algorithm.predict(titanic_test[predictors])

# Create a dataframe with requested columns for Kaggle submission
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived":predictions})

# Output results to csv file
submission.to_csv("titanic.csv", index=False)