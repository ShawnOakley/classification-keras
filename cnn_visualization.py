# Visualization using GraphViz
# Visualization using Quiver

from keras.utils import plot_model
plot_model(loaded_classifier, to_file='loaded_classifier.png')

from quiver engine import server
server.launch(loaded_classifier)