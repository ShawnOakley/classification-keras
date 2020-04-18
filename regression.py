# https://app.pluralsight.com/guides/regression-keras

# Import required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn

# Import necessary module
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

# Reading the Data and Performing Basic Data Checks
df = pd.read_csv('regressionexample.csv')
print(df.shape)
df.describe()

# Creating Arrays for the Features and the Response Variable
target_column = ['unemploy']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()

# Step 4 - Creating the Training and Test Datasets

X = df[predictors].values
y  = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

# Step 5 - Building the Deep Learning Regression Model
# Define model
model = Sequential()
model.add(Dense(500, input_dim=4, activation="relu")
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

# Step 4 - Creating the Training and Test Datasets
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train, y_train, epochs=20)

pred_train= model.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train)))

pred= model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred))) 