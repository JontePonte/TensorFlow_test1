
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

# Fix the data set index
word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0         # THe data will be padded so that every input has the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


print(decode_review(test_data[0]))
