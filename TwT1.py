
""" First example from Tech with Tim """

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

is_showimage = False
epochs_num = 5

""" Import data in a training set and a test set """

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Minimize data set to vary between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show image number 7 as example
if is_showimage:
    plt.imshow(train_images[7], cmap=plt.cm.binary)
    plt.show()


""" Create structure for the Neural Network """

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
# Set optimizer, loss function and correction metrics
model.compile(optimmizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=epochs_num)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc:", test_acc)
