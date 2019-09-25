
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define variables
learning_rate = 0.01
epochs = 200

# Create training data
n_samples = 30
train_x = np.linspace(0, 20, n_samples)
train_y = 3*train_x + 4*np.random.randn(n_samples)

plt.plot(train_x, train_y)
plt.show()