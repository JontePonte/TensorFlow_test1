
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

plt.imshow(train_images[7])
plt.show()

