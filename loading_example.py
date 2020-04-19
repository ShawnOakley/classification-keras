import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.models import model_from_json

json_file = open('iris_model.json', 'r')

json_model = json_file.read()

json_file.close()


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

loaded_model = model_from_json(json_model)

loaded_model.load_weights)'iris_model.h5')
print('loaded model from disk')

loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

loss, accuracy = loaded_model.evaluate(test_data_X, one_test_data_y, verbose=0)

