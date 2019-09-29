
""" All the imports """
import tensorflow as tf
from tensorflow import keras
import numpy as np

""" Variables """
epochs_nr = 40
batch_size_nr = 512
show_nr = 0

""" Import data in train and test sets"""
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Show the structure of the data set
print(train_data[show_nr])

""" Fix the data so it can be used in the NN """
word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0         # THe data will be padded so that every input has the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Do fancy stuff with data
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Pad the data with fancy tf function so that every text has the same length.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                       padding="post", maxlen=256)

# Function that decode input data to text
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# Print text instead of number coded data
print(decode_review(test_data[show_nr]))


""" Define the model structure """
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# Set optimizer and loss function
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

"""" Train the model and evaluate the results """
fit_model = model.fit(x_train, y_train, epochs=epochs_nr, batch_size=batch_size_nr,
                      validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

# Show a review, the label and the predicted label
test_review = test_data[show_nr]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: ", str(predict[show_nr]))
print("Actial: ", str(test_labels[show_nr]))
print(results)
