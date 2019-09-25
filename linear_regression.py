
# Apparently old and no longer working in newer versions of tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define variables
learning_rate = 0.01
epochs = 200

# Create training data with error
n_samples = 30
train_x = np.linspace(0, 20, n_samples)
train_y = 3*train_x + 4*np.random.randn(n_samples)

# Plot data
plt.plot(train_x, train_y)
plt.show()

# create tensorflow variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(),name="weights")
B = tf.Variable(np.random.randn(),name="bias")

# prediction function
pred = tf.add(tf.multiply(X, W), B)
# cost function
cost = tf.reduce_sum((pred - Y) ** 2)  / (2 * n_samples)
# Optimizing function
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sesh:
    sesh.run(init)

    for epoch in range(epochs):
        for x, y in zip(train_x, train_y):
            sesh.run(optimizer, feed_dict={X: x, Y: y})

        if not epochs % 20:
            c = sesh.run(cost, feed_dict={X: train_x, Y: train_y})
            w = sesh.run(W)
            b = sesh.run(B)
            print("epoch: {epoch:04d} c = {c:,4d} w = {w:,4f} b = {b:,4f}")