import pandas as pd

# Create dataframe from training data set
titanic = pd.read_csv("train.csv")

# Display passenger data for first five rows
print(titanic.head(5))

# Show summary stats for numeric columns
print(titanic.describe())