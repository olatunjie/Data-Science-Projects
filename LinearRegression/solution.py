import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Read the Smple Data
df = pd.read_csv('data/data.csv')

#print(df.head())

# Reshape the csv data
a = df['Hours'].values.reshape(-1,1)
b = df['Scores'].values

# Add a model
model = LinearRegression()

# Train the model
model.fit(a,b)

# Do a Prediction
b_pred = model.predict(a)

# Here I will Predict the score for 9.25 hours per day
hours = 9.5
predicted_score = model.predict([[hours]])

print(f'Predicted score if you study for {hours} hours/day is {predicted_score[0]:.2f}')