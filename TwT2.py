
""" All the imports """
import tensorflow as tf
from tensorflow import keras
import numpy as np

""" Import data in train and test sets"""
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Show the structure of the data set
if False:
    print(train_data[0])
word_index = imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.item()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

