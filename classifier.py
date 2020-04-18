# https://app.pluralsight.com/guides/classification-keras

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# import necessary modules
from sklearn.model_selection import train_test_split
import sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Read data and perform basic data checks
df = pd.read_csv('diabetes.csv')
print(df.shape)
df.describe()

# Creating Arrays for the Features and the Response Variable

# The first line of code creates an object of the target variable, while the second line of code gives the list of all the features after excluding the target variable, 'diabetes'.

# The third line does normalization of the predictors via scaling between 0 and 1. This is needed to eliminate the influence of the predictor's units and magnitude on the modelling process.

# The fourth line displays the summary of the normalized data. The target variable remains unchanged.
target_column = ['diabetes']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()

# Creating and Training and Test Datasets

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_stats=40)
print(X_train.shape); print(X_test.shape)

# Since our target variable represents a binary category which has been coded as numbers 0 and 1, we will have to encode it. We can easily achieve that using the "to_categorical" function from the Keras utilities package.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

# Define, Compile and Fit the Keras Classification Model
model = Sequential()
model.add(Dense(500, activation='relu', input_dims=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Build the model
model.fit(X_train, y_train, epochs=20)

pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

pred_test = model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

