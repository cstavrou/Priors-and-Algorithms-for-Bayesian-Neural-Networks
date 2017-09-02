# Create mnist small dataset
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def tabulate_proportions(labels):
    x = np.array(labels)
    unique, counts = np.unique(x, return_counts=True)
    proportions = counts/sum(counts)
    mat = np.asmatrix((unique, proportions)).T
    return mat

# Load the MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Select a small subset of the training data
y_train = mnist.train.labels[:2000,:]
x_train = mnist.train.images[:2000,:]

print('\n Train label proportions \n', tabulate_proportions(np.argmax(y_train, axis=1)))

# Select a small subset of test data
y_test = mnist.test.labels[:2000,:]
x_test = mnist.test.images[:2000,:]

print('\n Test label proportions \n', tabulate_proportions(np.argmax(y_test, axis=1)))

np.save('../data/x_train.npy', x_train)
np.save('../data/x_test.npy', x_test)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)