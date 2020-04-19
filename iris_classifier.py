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

