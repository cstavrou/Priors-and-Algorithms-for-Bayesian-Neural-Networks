from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import sys
import random
import os
import timeit

# ===============================================================================
# Data manipulation
# ===============================================================================

x_test = np.load('../data/x_test.npy')
x_train = np.load('../data/x_train.npy')
y_test = np.load('../data/y_test.npy')
y_train = np.load('../data/y_train.npy')


np.random.seed(seed=314159)           # random seed for reproducibility


N, D = np.shape(x_train)              # number of data points,  number of features.
K = np.shape(y_train)[1]              # number of classes.


# ===============================================================================
# Inputs
# ===============================================================================
batch_size = int(sys.argv[1])         # number of images in a minibatch.
n_hidden = int(sys.argv[2])           # number of hidden units 
n_layer = int(sys.argv[3])            # number of hidden layers
act_fun = str(sys.argv[4])            # activation function
optimiser = str(sys.argv[5])          # optimiser 
n_epoch = int(sys.argv[6])            # number of epochs for training
learn_r = float(sys.argv[7])          # learning rate for optimiser

# Allowed inputs

# 1. batch_size	(int <= 5000: N%batch_size = 0)
# 2. n_hidden	(int)
# 3. n_layer	(int: 1/2)
# 4. act_fun    (str: relu/tanh/softplus)
# 5. optimiser  (str: sgd/adam/momentum/nag)
# 6. n_epoch    (int > 0)
# 7. learn_r    (float in (0,1))


# ===============================================================================
# Functions
# ===============================================================================


# Error if the batch size does not divide the number of training examples exactly
#
assert(N%batch_size == 0), "Number of datapoints not divisible by batch size!"


# Initialise the parameters
#
def initialise_par(size):
	size_b = [size[len(size) - 1]]
	weights = tf.random_normal(size, stddev = 0.1)
	bias = tf.random_normal(size_b, stddev = 0.1)
	return tf.Variable(weights), tf.Variable(bias)


# For ease of the programming (see above error)
#
def batch_ind(x, n_epoch, batch_size):
	N = np.shape(x)[0]
	n_batch = int(N/batch_size)
	indices = np.zeros([n_epoch, n_batch, batch_size]).astype(int)
	for i in range(n_epoch):
		# Shuffle indices
		rand = random.sample(range(N), N)
		for j in range(n_batch):
			indices[i, j, :] = rand[j*batch_size:(j+1)*batch_size]
	return indices

# Define the neural network graph
#
if n_layer == 2:
	def build_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
		if act_fun == 'relu':
			h1 = tf.nn.relu_layer(x, W_0, b_0)
			h2 = tf.nn.relu_layer(h1, W_1, b_1)
		if act_fun == 'tanh':
			h1 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
			h2 = tf.nn.tanh(tf.matmul(h1, W_1) + b_1)
		if act_fun == 'softplus':
			h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
			h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
		y = tf.matmul(h2, W_2) + b_2
		return tf.reshape(y, [-1])

if n_layer == 1:
	def build_nn(x, W_0, b_0, W_1, b_1):
		if act_fun == 'relu':
			h1 = tf.nn.relu_layer(x, W_0, b_0)
		if act_fun == 'tanh':
			h1 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
		if act_fun == 'softplus':
			h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
		y = tf.matmul(h1, W_1) + b_1
		return tf.reshape(y, [-1])

# Path where files will be stored
#
path =  ('../saved/' + str(n_hidden) +'units/'+ str(n_layer) + 'l_' + str(n_epoch) + 'rep/' + 
	optimiser + '/learning_rate_' + str(learn_r).replace('.',''))
if not os.path.exists(path):
  os.makedirs(path)


# ===============================================================================
# Build the graph
# ===============================================================================

# Input data
x = tf.placeholder("float32", shape=[None, D])
# Target output value
y_ = tf.placeholder("float32", shape=[None])

# Assigning the parameters
#
if n_layer == 2:
	W_0, b_0 = initialise_par([D, n_hidden])
	W_1, b_1 = initialise_par([n_hidden, n_hidden])
	W_2, b_2 = initialise_par([n_hidden, K])
	y = build_nn(x, W_0, b_0, W_1, b_1, W_2, b_2)
if n_layer == 1:
	W_0, b_0 = initialise_par([D, n_hidden])
	W_1, b_1 = initialise_par([n_hidden, K])
	y = build_nn(x, W_0, b_0, W_1, b_1)

# Compute the mse 
mse = tf.reduce_mean(tf.square(tf.subtract(y, y_)))

# Define the optimiser function (used when running forward and backpropagation)
if optimiser == 'sgd':
	fbprop = tf.train.GradientDescentOptimizer(learn_r).minimize(mse)
if optimiser == 'adam':
	# Using default options for Adam
	fbprop = tf.train.AdamOptimizer(learn_r).minimize(mse)
if optimiser == 'momentum':
	m = 0.1
	if len(sys.argv) == 9:
		m = float(sys.argv[8])
	fbprop = tf.train.MomentumOptimizer(learning_rate=learn_r, 
		momentum=m).minimize(mse)
if optimiser == 'nag':
	m = 0.1
	if len(sys.argv) == 9:
		m = float(sys.argv[8])
	fbprop = tf.train.MomentumOptimizer(learning_rate=learn_r, 
		momentum=m, use_nesterov=True).minimize(mse)

indices_batch = batch_ind(x_train, n_epoch, batch_size)
n_batch = int(N/batch_size)

# Training 
test_mse, train_mse = [], []
y_train = np.reshape(y_train, [-1])
y_test = np.reshape(y_test, [-1])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training
	total = timeit.default_timer()

	for i in range(n_epoch):
		for j in range(n_batch):
			start = timeit.default_timer()
			ind = indices_batch[i, j, :]
			batch_x = x_train[ind, :]
			batch_y = y_train[ind]
			fbprop.run(feed_dict={x: batch_x, y_: batch_y})
			elapsed = timeit.default_timer() - start
			total = total + elapsed
		if (i+1)%20==0:
			tmp_mse = sess.run(mse, feed_dict={x: x_test, y_: y_test})
			test_mse.append(tmp_mse)
			train_mse.append(sess.run(mse, feed_dict={x: x_train, y_: y_train}))
			print('Epoch: ', i+1, ' -- MSE: ', tmp_mse)

	# Final mse
	mse_final = sess.run(mse, feed_dict={x: x_test, y_: y_test})
	print('Final test mse: ', mse_final)

	# Graph fitted/learnt function
	x_in = np.linspace(-2.5, 4, 400)
	y_hat = sess.run(y, feed_dict={x: np.reshape(x_in,[-1, 1])})

	fig, ax = plt.subplots(2)
	ax[0].plot(x_in, y_hat, color='black') 
	ax[0].scatter(x_test[:, 0], y_test)
	ax[0].set_xlabel('Input (x)')
	ax[0].set_ylabel('Output (y)')
	ax[0].set_title('Trained function')
	ax[1].plot(20*np.arange(n_epoch/20), test_mse, label='Test')
	ax[1].plot(20*np.arange(n_epoch/20), train_mse, color='orange', label='Train') 
	ax[1].set_xlabel('Epoch', fontsize=14)
	ax[1].set_ylabel('Test MSE', fontsize=14)
	ax[1].legend(loc='upper right', prop={'size': 14})
	ax[1].set_title('Learning curves ')
	if n_layer == 1:
		add_fig = ' hidden layer ('
	else:
		add_fig = ' hidden layers ('
	fig.suptitle(str(n_layer) + add_fig + str(n_hidden) + ' hidden units) - ' + str(optimiser) + '(LR = ' + str(learn_r) + ')')
	plt.subplots_adjust(hspace=0.4)
	plt.savefig(path + '/activation_' + act_fun + '_train_and_prediction_plots.png')
	plt.close(fig)

# Save the MSE list
np.save(path +'/activation_' + act_fun + '_test_mse_during_training.npy', test_mse)

print('Total time elapsed (seconds): ',total)
info = ['Total algorithm time (seconds) -- ' + str(total), 'Batch size -- ' + str(batch_size), 
'Final MSE -- ' + str(mse_final)]

name = path + '/activation_' + act_fun + '_info_file.csv'
# Save the info file
np.savetxt(name, info, fmt='%s' , delimiter=',')