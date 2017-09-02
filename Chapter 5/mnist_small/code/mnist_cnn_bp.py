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
N, D = np.shape(x_train)              # number of data points,  number of features.
K = np.shape(y_train)[1]              # number of classes.

C = 10
F = 3

# ===============================================================================
# Inputs
# ===============================================================================
batch_size = int(sys.argv[1])         # number of images in a minibatch.
n_hidden = int(sys.argv[2])           # number of hidden units 
act_fun = str(sys.argv[3])            # activation function
optimiser = str(sys.argv[4])          # optimiser 
n_epoch = int(sys.argv[5])            # number of epochs for training
learn_r = float(sys.argv[6])          # learning rate for optimiser



# Allowed inputs

# 1. batch_size	(int <= 5000: N%batch_size = 0)
# 2. n_hidden	(int)
# 3. act_fun    (str: relu/tanh/softplus)
# 4. optimiser  (str: sgd/adam/momentum/nag)
# 5. n_epoch    (int > 0)
# 6. learn_r    (float in (0,1))

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

def conv_layer(x, W, b):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') + b

def hidden_layer(X, W, b):
	if act_fun == 'relu':
		return tf.nn.relu(tf.matmul(X, W) + b)
	if act_fun == 'softplus':
		return tf.nn.softplus(tf.matmul(X, W) + b)
	if act_fun == 'tanh':
		return tf.nn.tanh(tf.matmul(X, W) + b)
    
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

def build_nn(x, W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3):
    x = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x, W_0, b_0)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_layer(pool1, W_1, b_1)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    fc = tf.reshape(pool2, [-1, 7*7*C])
    h = hidden_layer(fc, W_2, b_2)
    return tf.matmul(h, W_3) + b_3

# ===============================================================================
# Build the graph
# ===============================================================================

# Input data
x = tf.placeholder("float32", shape=[None, D])
# Target output value
y_ = tf.placeholder("float32", shape=[None, K])

W_0, b_0 = initialise_par([F, F, 1, C])
W_1, b_1 = initialise_par([F, F, C, C])
W_2, b_2 = initialise_par([7*7*C, n_hidden])
W_3, b_3 = initialise_par([n_hidden, K])
y = build_nn(x, W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3)

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
	if len(sys.argv) == 8:
		m = float(sys.argv[7])
	fbprop = tf.train.MomentumOptimizer(learning_rate=learn_r, 
		momentum=m).minimize(cross_ent)
if optimiser == 'nag':
	m = 0.1
	if len(sys.argv) == 8:
		m = float(sys.argv[7])
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
			fbprop.run(feed_dict={x: batch_x, y_: batch_y})
			elapsed = timeit.default_timer() - start
			total = total + elapsed
		if (i+1)%20==0 or i == 0:
			tmp_acc = sess.run(acc, feed_dict={x: x_test, y_: y_test})
			test_acc.append(tmp_acc)
			train_acc.append(sess.run(acc, feed_dict={x: x_train, y_: y_train}))
			print('Epoch: ', i+1, ' -- Test accuracy = ', tmp_acc)

	# Final Accuracy
	acc_final = sess.run(acc, feed_dict={x: x_test, y_: y_test})
	print('Final test accuracy = ', acc_final)

path =  ('../saved/' + str(n_hidden) +'units/'+ 'conv_net_' + str(n_epoch) + 'rep/' + 
	optimiser + '/learning_rate_' + str(learn_r).replace('.','_'))
if not os.path.exists(path):
  os.makedirs(path)


# Plot the learning curve
fig, ax = plt.subplots(1)
xx = 20*np.arange(n_epoch/20 + 1)
ax.plot(xx, test_acc, label='Test')
ax.plot(xx, train_acc, color='orange', label='Train')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.legend(loc='lower right', prop={'size': 14})
fig.suptitle('CNN 2x(conv & maxpool) + Hidden layer (' + str(n_hidden) + ' hidden units) - ' + str(optimiser) + '(LR = ' + str(learn_r) + ')')
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