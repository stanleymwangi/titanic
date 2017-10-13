# Import libraries for analysis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold # helper for cross validation

# Create dataframe from clean training data set
titanic = pd.read_csv("clean_train.csv")

# Columns for target prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialise algorithm class
linreg_algorithm = LinearRegression()

# Make cross-validation folds for titanic data set
kf = KFold(n_splits=3, random_state=1)  # setting random_state ensures splits stay the same on each run

predictions = []
for train, test in kf.split(titanic): # kf.split produces row indices for training and testing

    # Predictors for training the algorithm from the training fold
    train_predictors = (titanic[predictors].iloc[train,:])

    # Target for training the algorithm from the training fold
    train_target = titanic["Survived"].iloc[train]

    # Train algorithm using predictors and target
    linreg_algorithm.fit(train_predictors, train_target)

    # Apply trained algorithm to test fold to make predictions
    test_predictions = linreg_algorithm.predict(titanic[predictors].iloc[test,:])

    # Add resulting predictions
    predictions.append(test_predictions)

# Evaluate error
# Combine numpy arrays which hold our separate predictions into one column
predictions = np.concatenate(predictions, axis=0)

# Map predictions to 0 or 1
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

# Error metric is the percentage of correct predictions
accuracy = round(np.sum(predictions == titanic["Survived"]) / titanic.shape[0] * 100, 1)

print(accuracy)