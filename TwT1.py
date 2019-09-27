
""" First example from Tech with Tim """

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras


""" Import data in a training set and a test set """

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Minimize data set to vary between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show image number 7 as example
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

