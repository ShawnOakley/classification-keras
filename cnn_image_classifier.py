import numpy as numpy
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Construct model

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (50,50,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metric = ['accuracy'])

classifier.summary()

# Feed in data
# Rescales the rgb value
# Randomly shears the image
# Randomyl zooms
# Randomly flips
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)



# Only rescale the test data.  No other augmentation
test_datagen =  ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
    target_size = (50,50),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
    target_size = (50,50),
    batch_size=32,
    class_mode='binary')

x, y = training_set.next()
for i in range(0,1):
    random_image = x[i]
    plt.imshow(random_image)
    plt.show()

classifier.fit_generator(training_set,
    steps_per_epoch = 8000,
    epochs = 25,
    validation_data = test_set,
    validation_steps = 2000
)
