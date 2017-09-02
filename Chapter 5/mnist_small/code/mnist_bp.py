from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import sys
import random
import os
import timeit
import cv2
from sklearn.metrics import confusion_matrix

# ===============================================================================
# Data manipulation
# ===============================================================================

x_test = np.load('../data/x_test.npy')
x_train = np.load('../data/x_train.npy')
y_test = np.load('../data/y_test.npy')
y_train = np.load('../data/y_train.npy')


np.random.seed(seed=314159)
N = np.shape(x_train)[0]              # number of data points,  number of features.
K = np.shape(y_train)[1]              # number of classes.

D = int(14**2)                        # dimensions of the reduced image

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


# Initialise the parameters (randomly sampling from a normal distribution)
#
def initialise_par(size):
	size_b = [size[len(size) - 1]]
	weights = tf.random_normal(size, stddev = 0.1)
	bias = tf.random_normal(size_b, stddev = 0.1)
	return tf.Variable(weights), tf.Variable(bias)

def resize(images):
	im = np.reshape(images, [-1,28,28])
	n = np.shape(im)[0]
	reduced_im = np.zeros([n, 14, 14])
	for ind in range(n):
		reduced_im[ind,:,:] = cv2.resize(im[ind,:,:], (14, 14))
	grey_im = (0.1 < reduced_im).astype('float32')
	return np.reshape(grey_im, [-1, 14*14])


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

# Define the neural network
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
		return tf.matmul(h2, W_2) + b_2

if n_layer == 1:
	def build_nn(x, W_0, b_0, W_1, b_1):
		if act_fun == 'relu':
			h1 = tf.nn.relu_layer(x, W_0, b_0)
		if act_fun == 'tanh':
			h1 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
		if act_fun == 'softplus':
			h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
		return tf.matmul(h1, W_1) + b_1


# ===============================================================================
# Build the graph
# ===============================================================================

# Input data
x = tf.placeholder("float32", shape=[None, D])
# Target output value
y_ = tf.placeholder("float32", shape=[None, K])


if n_layer == 2:
	W_0, b_0 = initialise_par([D, n_hidden])
	W_1, b_1 = initialise_par([n_hidden, n_hidden])
	W_2, b_2 = initialise_par([n_hidden, K])
	y = build_nn(x, W_0, b_0, W_1, b_1, W_2, b_2)

if n_layer == 1:
	W_0, b_0 = initialise_par([D, n_hidden])
	W_1, b_1 = initialise_par([n_hidden, K])
	y = build_nn(x, W_0, b_0, W_1, b_1)

# Predictions from the NN
# y_pred = tf.reshape(tf.argmax(tf.nn.softmax(y), 1), [-1])

# Compute the cross entropy
cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# Compute the accuracy 
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

# Optimiser function: runs forward and backpropagation
if optimiser == 'sgd':
	fbprop = tf.train.GradientDescentOptimizer(learn_r).minimize(cross_ent)
if optimiser == 'adam':
	# Using default options for Adam
	fbprop = tf.train.AdamOptimizer(learn_r).minimize(cross_ent)
if optimiser == 'momentum':
	m = 0.1
	if len(sys.argv) == 9:
		m = float(sys.argv[8])
	fbprop = tf.train.MomentumOptimizer(learning_rate=learn_r, 
		momentum=m).minimize(cross_ent)
if optimiser == 'nag':
	m = 0.1
	if len(sys.argv) == 9:
		m = float(sys.argv[8])
	fbprop = tf.train.MomentumOptimizer(learning_rate=learn_r, 
		momentum=m, use_nesterov=True).minimize(cross_ent)

indices_batch = batch_ind(x_train, n_epoch, batch_size)
n_batch = int(N/batch_size)

# Training 
test_acc, train_acc = [], []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training
	total = timeit.default_timer()

	for i in range(n_epoch):
		for j in range(n_batch):
			start = timeit.default_timer()
			ind = indices_batch[i, j, :]
			batch_x = x_train[ind, :]
			batch_y = y_train[ind, :]
			fbprop.run(feed_dict={x: resize(batch_x), y_: batch_y})
			elapsed = timeit.default_timer() - start
			total = total + elapsed
		if (i+1)%20==0 or i == 0:
			tmp_acc = sess.run(acc, feed_dict={x: resize(x_test), y_: y_test})
			test_acc.append(tmp_acc)
			train_acc.append(sess.run(acc, feed_dict={x: resize(x_train), y_: y_train}))
			print('Epoch: ', i+1, ' -- Test accuracy = ', tmp_acc)

	# Final Accuracy
	acc_final = sess.run(acc, feed_dict={x: resize(x_test), y_: y_test})
	print('Final test accuracy = ', acc_final)

path =  ('../saved/' + str(n_hidden) +'units/'+ str(n_layer) + 'l_' + str(n_epoch) + 'rep/' + 
	optimiser + '/learning_rate_' + str(learn_r).replace('.','_'))
if not os.path.exists(path):
  os.makedirs(path)


# Plot the learning curve
fig, ax = plt.subplots(1)
xx = 20*np.arange(n_epoch/20 + 1)
ax.plot(xx, test_acc, color='blue', label='Test')
ax.plot(xx, train_acc, color='orange', label='Train')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.legend(loc='lower right', prop={'size': 14})
plt.savefig(path + '/activation_' + act_fun + '_training_curve.png')
plt.close(fig)

# Save the accuracy during training
np.save(path +'/activation_' + act_fun + '_test_acc.npy', test_acc)
np.save(path +'/activation_' + act_fun + '_train_acc.npy', train_acc)

print('Total time elapsed (seconds): ',total)
info = ['Total algorithm time (seconds) -- ' + str(total), 'Batch size -- ' + str(batch_size), 
'Final Test Accuracy -- ' + str(acc_final)]

name = path + '/activation_' + act_fun + '_info_file.csv'
np.savetxt(name, info, fmt='%s' , delimiter=',')