# https://app.pluralsight.com/course-player?clipId=28da0493-613d-47c9-8ebb-d26d942e02bf
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

iris = sns.load_dataset('iris')
iris.head()

iris.shape

X = iris.values(1, :4)
y = iris.values(:, 4)

train_data_X, test_data_X, train_data_y, test_data_y = \
    train_test_split(X, y, train_size=0.5, test_size=0.5, random_state-0)

def one_hot_encoder(array):
    unique_values, indices = np.unique(array, return_inverse=True)
    one_hot_encoded_data = np_utils.to_categorical(indices, len(unique_values))
    return one_hot_encoded_data

one_train_data_y = one_hot_encoder(train_data_y)
one_test_data_y = one_hot_encoder(test_data_y)

model = Sequential()

model.add(Dense(16, input_shape=(4,), name='Input_Layer', activation='relu'))

model.add(Dense(3, name='Output_Layer', activation='softmax'))

model.summmary()

model.compile(optimizer='adam', loss='categorical_crossentrpy', metric=['accuracy'])

model.fit(train_data_X, one_train_data_y, epochs=10, batch_size=2, verbose=2)

loss, accuracy = model.evaluate(test_data_X, one_test_data_y, verbose=0)

