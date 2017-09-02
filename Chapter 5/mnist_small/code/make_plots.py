# ======================================================================================================================
# Script to make the plots for the small MNIST dataset model
# ======================================================================================================================

import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit
import random
import cv2
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Load the MNIST data
X_test = np.load('../data/x_test.npy')
Y_test = np.load('../data/y_test.npy')
X_train = np.load('../data/x_train.npy')
Y_train = np.load('../data/y_train.npy')
K = np.shape(Y_train)[1]

n_hidden = int(sys.argv[1])  				# number of hidden units
method = str(sys.argv[2])                   # method used (hmc/sghmc)
prior = str(sys.argv[3])                    # prior dist (T/normal/laplace)
n_samp = int(sys.argv[4])                   # number of samples for HMC.
mod = str(sys.argv[5]) 				        # 1l/2l/conv_net
plt_no = int(sys.argv[6])                   # last file number (see folder and number for last file = n_iter - 1)
trace_plt = str(sys.argv[7])                # True/False (whether to make the trace plot for the weights)
prob_trace = str(sys.argv[8])				# True/False (whether to make a probability trace plot)
mar_plots = str(sys.argv[9])                # True/False (whether to make the marginal dist plots)
skip_first = str(sys.argv[10])              # True/False (whether to skip the burnin iteration when plotting the accuracy curve)
samp100 = str(sys.argv[11])                 # True/False (whether to use the additional per 100 samples collected)
if samp100 == 'True':
	spac_samp = int(sys.argv[12])           # Every how many samples to pick a sample (the samples are already spaced by 100)
	prop_burn = float(sys.argv[13])         # Proportion of samples to burn (samples to burnin after the initial burned phase)
if str(sys.argv[3]) == 'T':
	df = float(sys.argv[14])                # degrees of freedom for the T-distribution prior

if mod == 'conv_net':
	D = np.shape(X_train)[1]
else:
	D = int(14**2)

# Reset tf graph
tf.reset_default_graph()

# Path where to find the files and save the plots
#
path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_samp) + 'rep/' + 
	method + '/' + prior)
if prior != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_samp) + 
		'rep/' + method + '/' + prior)
else:
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_samp) + 
		'rep/' + method + '/' + prior + '_' + str(df).replace('.','_'))

# Make the probability trace plot
#
if prob_trace == 'True':
	for i in range(1, plt_no+1):
		if i == 1:
			prob_trace = np.load(path + '/prob_trace' + str(i) + '.npy')
		else:
			prob_trace = np.concatenate([prob_trace, 
				np.load(path + '/prob_trace' + str(i) + '.npy')] , 1)
		
	fig, ax = plt.subplots(5,2)
	for i in range(2):
		for j in range(5):
			tmp = prob_trace[5*i+j, :]
			ax[j, i].plot(10*np.arange(len(tmp)), tmp)
			ax[j, i].set_xlabel('Iteration') 
	plt.subplots_adjust(hspace=0.5)
	plt.subplots_adjust(wspace=0.7)
	plt.savefig(path + '/prob_traceplot' + str(plt_no) + '.png')
	plt.close(fig)

if skip_first != 'True':
	acc = pd.read_csv(path + '/test_acc.csv', header=None)

# Make the accuracy plots
#
for i in range(1, plt_no+1):
	tmp = pd.read_csv(path + '/test_acc' + str(i) + '.csv', header=None)
	if skip_first != 'True' or i > 1:
		acc = np.concatenate([acc, tmp])
	else:
		acc = tmp

fig, ax = plt.subplots(1)
ax.plot(50*np.arange(len(acc)), acc)
ax.set_xlabel('Iteration') 
ax.set_ylabel('Accuracy')
if skip_first == 'True':
	plt.savefig(path + '/acc_plot_skip_first.png')
else:
	plt.savefig(path + '/acc_plot.png')
plt.close(fig)

# Make the weight trace plot
#
if trace_plt == 'True':
	if mod == 'conv_net':
		w0 = np.load(path + '/traceplot_w0.npy')
		w1 = np.load(path + '/traceplot_w1.npy')
		w3 = np.load(path + '/traceplot_w3.npy')
		fig, ax = plt.subplots(3)
		for i in range(np.shape(w0)[0]):
			if i == 0:
				w0_new = w0[i, :, :]
				w1_new = w1[i, :, :]
				w3_new = w3[i, :, :]
			else: 
				w0_new = np.concatenate([w0_new, w0[i, :, :]], 1)
				w1_new = np.concatenate([w1_new, w1[i, :, :]], 1)
				w3_new = np.concatenate([w3_new, w3[i, :, :]], 1)
		for i in range(np.shape(w0_new)[0]):
			ax[0].plot(w0_new[i, :])
		for i in range(np.shape(w1_new)[0]):
			ax[1].plot(w1_new[i, :])
		for i in range(np.shape(w3_new)[0]):
			ax[2].plot(w3_new[i, :])
		ax[0].set_xlabel('Iteration') 
		ax[1].set_xlabel('Iteration')
		ax[2].set_xlabel('Iteration') 
	if mod == '2l':
		w0 = np.load(path + '/traceplot_w0.npy')
		w1 = np.load(path + '/traceplot_w1.npy')
		w2 = np.load(path + '/traceplot_w2.npy')
		fig, ax = plt.subplots(3)
		for i in range(np.shape(w0)[0]):
			if i == 0:
				w0_new = w0[i, :, :]
				w1_new = w1[i, :, :]
				w2_new = w2[i, :, :]
			else: 
				w0_new = np.concatenate([w0_new, w0[i, :, :]], 1)
				w1_new = np.concatenate([w1_new, w1[i, :, :]], 1)
				w2_new = np.concatenate([w2_new, w2[i, :, :]], 1)
		for i in range(np.shape(w0_new)[0]):
			ax[0].plot(w0_new[i, :])
		for i in range(np.shape(w1_new)[0]):
			ax[1].plot(w1_new[i, :])
		for i in range(np.shape(w2_new)[0]):
			ax[2].plot(w2_new[i, :])
		ax[0].set_xlabel('Iteration') 
		ax[1].set_xlabel('Iteration')
		ax[2].set_xlabel('Iteration') 
	if mod == '1l':
		w0 = np.load(path + '/traceplot_w0.npy')
		w1 = np.load(path + '/traceplot_w1.npy')
		fig, ax = plt.subplots(2)
		for i in range(np.shape(w0)[0]):
			if i == 0:
				w0_new = w0[i, :, :]
				w1_new = w1[i, :, :]
			else: 
				w0_new = np.concatenate([w0_new, w0[i, :, :]], 1)
				w1_new = np.concatenate([w1_new, w1[i, :, :]], 1)
		for i in range(np.shape(w0_new)[0]):
			ax[0].plot(w0_new[i, :])
		for i in range(np.shape(w1_new)[0]):
			ax[1].plot(w1_new[i, :])
		ax[0].set_xlabel('Iteration') 
		ax[1].set_xlabel('Iteration')
	plt.subplots_adjust(hspace=0.3) 
	plt.savefig(path + '/weight_traceplot.png')
	plt.close(fig)


# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(Y_test,axis=1)
Y_train = np.argmax(Y_train, axis=1)

# Halves the size of the images from (28x28) -> (14x14)
#
def resize(images):
	im = np.reshape(images, [-1,28,28])
	n = np.shape(im)[0]
	reduced_im = np.zeros([n, 14, 14])
	for ind in range(n):
		reduced_im[ind,:,:] = cv2.resize(im[ind,:,:], (14, 14))
	grey_im = (0.1 < reduced_im).astype('float32')
	return np.reshape(grey_im, [-1, 14*14])

if mod == '1l':
	def pred_nn(x, W_0, b_0, W_1, b_1):
	    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
	    o = tf.nn.softmax(tf.matmul(h, W_1) + b_1)
	    return tf.reshape(tf.argmax(o, 1), [-1])
    
	def probs_nn(x, W_0, b_0, W_1, b_1):
		h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
		out_probs = tf.nn.softmax(tf.matmul(h, W_1) + b_1)
		return tf.reshape(out_probs, [-1, K])
    
    # Build predictive graph
	x_pred = tf.placeholder(tf.float32, [None, None])
	ww0 = tf.placeholder(tf.float32, [None, None])
	ww1 = tf.placeholder(tf.float32, [None, None])
	bb0 = tf.placeholder(tf.float32, [None])
	bb1 = tf.placeholder(tf.float32, [None])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1)
	# Build predictive graph for class probabilities
	prob_out = probs_nn(x_pred, ww0, bb0, ww1, bb1)

if mod == '2l':
	def pred_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
	    h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
	    h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
	    o = tf.nn.softmax(tf.matmul(h2, W_2) + b_2)
	    return tf.reshape(tf.argmax(o, 1), [-1])

	def probs_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
		h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
		h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
		o = tf.nn.softmax(tf.matmul(h2, W_2) + b_2)
		return tf.reshape(o, [-1, K])

	# Build predictive graph
	x_pred = tf.placeholder(tf.float32, [None, None])
	ww0 = tf.placeholder(tf.float32, [None, None])
	ww1 = tf.placeholder(tf.float32, [None, None])
	ww2 = tf.placeholder(tf.float32, [None, None])
	bb0 = tf.placeholder(tf.float32, [None])
	bb1 = tf.placeholder(tf.float32, [None])
	bb2 = tf.placeholder(tf.float32, [None])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2)
	# Build predictive graph for class probabilities
	prob_out = probs_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2)

if mod == 'conv_net':
	C = 10
	F = 3
	def conv_layer(x, W, b):
	    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') + b

	def hidden_layer(X, W, b):
	    return tf.nn.softplus(tf.matmul(X, W) + b)

	def pred_nn(x, W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3):
	    x = tf.reshape(x, [-1, 28, 28, 1])
	    conv1 = conv_layer(x, W_0, b_0)
	    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	    conv2 = conv_layer(pool1, W_1, b_1)
	    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	    fc = tf.reshape(pool2, [-1, 7*7*C])
	    h = hidden_layer(fc, W_2, b_2)
	    out = tf.nn.softmax(tf.matmul(h, W_3) + b_3)
	    return tf.reshape(tf.argmax(out, 1), [-1])
	def probs_nn(x, W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3):
	    x = tf.reshape(x, [-1, 28, 28, 1])
	    conv1 = conv_layer(x, W_0, b_0)
	    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	    conv2 = conv_layer(pool1, W_1, b_1)
	    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	    fc = tf.reshape(pool2, [-1, 7*7*C])
	    h = hidden_layer(fc, W_2, b_2)
	    out_probs = tf.nn.softmax(tf.matmul(h, W_3) + b_3)
	    return tf.reshape(out_probs, [-1, K])

	# Build predictive graph
	x_pred = tf.placeholder(tf.float32, [None, None])
	ww0 = tf.placeholder(tf.float32, [None, None, None, None])
	ww1 = tf.placeholder(tf.float32, [None, None, None, None])
	ww2 = tf.placeholder(tf.float32, [None, None])
	ww3 = tf.placeholder(tf.float32, [None, None])
	bb0 = tf.placeholder(tf.float32, [None])
	bb1 = tf.placeholder(tf.float32, [None])
	bb2 = tf.placeholder(tf.float32, [None])
	bb3 = tf.placeholder(tf.float32, [None])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2, ww3, bb3)
	prob_out = probs_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2, ww3, bb3)

def mean_acc(Y_true, Y_hat):
    acc = Y_true == Y_hat
    return np.mean(acc)

# Using the files of samples saved once every 100 iterations of HMC or SGHMC
#
if samp100 == 'True':
	name = '_n100'
else:
	name = ''

qW0_smp = np.load(path + '/qW0_samp' + name + '.npy')
qW1_smp = np.load(path + '/qW1_samp' + name + '.npy')
qb0_smp = np.load(path + '/qb0_samp' + name + '.npy')
qb1_smp = np.load(path + '/qb1_samp' + name + '.npy')

if mod == '2l' or mod == 'conv_net':
	qW2_smp = np.load(path + '/qW2_samp' + name + '.npy')
	qb2_smp = np.load(path + '/qb2_samp' + name + '.npy')
if mod == 'conv_net':
	qW3_smp = np.load(path + '/qW3_samp' + name + '.npy')
	qb3_smp = np.load(path + '/qb3_samp' + name + '.npy')

# Spaces out the samples taken every 100 and removes the appropriate burnin
#
if samp100 == 'True':
	aa = np.shape(qW0_smp)[0]
	burnin = int(prop_burn*aa)
	if mod == '1l' or mod == '2l':
		qW0_smp, qW1_smp, qb0_smp, qb1_smp = (qW0_smp[burnin:aa:spac_samp, :, :], qW1_smp[burnin:aa:spac_samp, :, :],
			qb0_smp[burnin:aa:spac_samp, :], qb1_smp[burnin:aa:spac_samp, :])
	if mod == '2l' or mod == 'conv_net':
		qW2_smp, qb2_smp = qW2_smp[burnin:aa:spac_samp, :, :], qb2_smp[burnin:aa:spac_samp, :]
	if mod == 'conv_net':
		qW0_smp, qW1_smp, qW3_smp = (qW0_smp[burnin:aa:spac_samp, :, :, :, :], 
			qW1_smp[burnin:aa:spac_samp, :, :, :, :], qW3_smp[burnin:aa:spac_samp, :, :])
		qb0_smp, qb1_smp, qb3_smp = (qb0_smp[burnin:aa:spac_samp, :], 
			qb1_smp[burnin:aa:spac_samp, :], qb3_smp[burnin:aa:spac_samp, :])

# Final prediction
acc_final = []
conf_mat = np.zeros([np.shape(qW0_smp)[0], K, K])
probs = np.zeros([np.shape(Y_test)[0], K])

with tf.Session() as sess:
	# Initialise all the vairables in the session.
	sess.run(tf.global_variables_initializer())

	for i in range(np.shape(qW0_smp)[0]):
		if mod == '1l':
			pred, p_out = sess.run([y_pred, prob_out], feed_dict={x_pred: resize(X_test), ww0: qW0_smp[i, :, :],
					bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
			if i+1 == np.shape(qW0_smp)[0]:
				point_est = sess.run(y_pred, feed_dict={x_pred: resize(X_test), ww0: qW0_smp.mean(axis=0),
					bb0: qb0_smp.mean(axis=0), ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0)})
		if mod == '2l': 
			pred, p_out = sess.run([y_pred, prob_out], feed_dict={x_pred: resize(X_test), ww0: qW0_smp[i, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :],
				ww2: qW2_smp[i, :, :], bb2: qb2_smp[i, :]})
			if i+1 == np.shape(qW0_smp)[0]:
				point_est = sess.run(y_pred, feed_dict={x_pred: resize(X_test), ww0: qW0_smp.mean(axis=0), 
					bb0: qb0_smp.mean(axis=0), ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0), 
					ww2: qW2_smp.mean(axis=0), bb2: qb2_smp.mean(axis=0)})
		if mod == 'conv_net':
			pred, p_out = sess.run([y_pred, prob_out], feed_dict={x_pred: X_test, ww0: qW0_smp[i, :, :, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :, :, :], bb1: qb1_smp[i, :],
				ww2: qW2_smp[i, :, :], bb2: qb2_smp[i, :], ww3: qW3_smp[i, :, :], bb3: qb3_smp[i, :]})
			if i+1 == np.shape(qW0_smp)[0]:
				point_est = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: qW0_smp.mean(axis=0),
					bb0: qb0_smp.mean(axis=0), ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0), 
					ww2: qW2_smp.mean(axis=0), bb2: qb2_smp.mean(axis=0), 
					ww3: qW3_smp.mean(axis=0), bb3: qb3_smp.mean(axis=0)})
		probs = probs + p_out
		acc_final.append(mean_acc(Y_test, pred))
		tmp_conf = confusion_matrix(Y_test, pred)
		# Express confussion matrix as probabilities
		conf_mat[i, :, :] = (tmp_conf.T/np.sum(tmp_conf, axis=1)).T
	conf_point_est = confusion_matrix(Y_test, point_est)
	y_hat = np.reshape(np.argmax(probs, axis=1), [-1])
	fin_acc = mean_acc(Y_test, y_hat)
	print('Final prediction accuracy = ', fin_acc, ' +/- ', str(np.std(acc_final)))
	print('Total number of samples used = ', str(np.shape(qW0_smp)[0]))
	print('Point estimate accuracy = ', str(mean_acc(Y_test, point_est)))

	# Graph of drawn confussion matrices from the posterior
	conf_mat_mean = confusion_matrix(Y_test, y_hat)

	# Confusion matrix of the MC estimate (expressed as probabilities)
	#
	fig, ax = plt.subplots(1)
	cbar = ax.imshow((conf_mat_mean.T/np.sum(conf_mat_mean, axis=1)).T, cmap=plt.cm.gnuplot, interpolation='none')
	ax.set_xticks(np.arange(0, 9, 2))
	ax.grid(False)
	fig.colorbar(cbar, orientation='vertical')
	ax.set_xlabel('Predicted', fontsize=14) 
	ax.set_ylabel('Actual', fontsize=14)
	plt.savefig(path + '/predictive_conf_matrix_new.png')
	plt.close(fig)

	# Confussion matrix for the point estimate
	#
	fig, ax = plt.subplots(1)
	cbar = ax.imshow((conf_point_est.T/np.sum(conf_point_est, axis=1)).T, cmap=plt.cm.gnuplot, interpolation='none')
	ax.set_xticks(np.arange(0, 9, 2))
	ax.grid(False)
	fig.colorbar(cbar, orientation='vertical')
	ax.set_xlabel('Predicted', fontsize=14) 
	ax.set_ylabel('Actual', fontsize=14)
	plt.savefig(path + '/point_estimate_conf_matrix_new.png')
	plt.close(fig)

	# Plot the standard deviation of the probability confussion matrix (of the MC estimate)
	#
	fig, ax = plt.subplots(1)
	cbar = ax.imshow(np.round(np.std(conf_mat, axis=0), decimals=4), cmap=plt.cm.gnuplot, interpolation='none')
	ax.set_xticks(np.arange(0, 9, 2))
	ax.grid(False)
	fig.colorbar(cbar, orientation='vertical')
	ax.set_xlabel('Predicted', fontsize=14) 
	ax.set_ylabel('Actual', fontsize=14)
	plt.subplots_adjust(hspace=0.2)
	plt.savefig(path + '/predictive_conf_matrix_std_new.png')
	plt.close(fig)


# Make marginal distribution plots 
#
if mod == '2l' or mod == '1l':
	ii0 = random.sample(range(D), 4)
	jj0 = random.sample(range(n_hidden), 3)
	ii1 = random.sample(range(n_hidden), 4)
	jj1 = random.sample(range(K), 3)

	if mod == '2l':
		fig, ax = plt.subplots(3)
		for i in range(4):
			for j in range(3):
				sns.distplot(qW0_smp[:, ii0[i], jj0[j]], hist=False, rug=False, ax=ax[0])
				sns.distplot(qW1_smp[:, ii1[i], jj0[j]], hist=False, rug=False, ax=ax[1])
				sns.distplot(qW2_smp[:, ii1[i], jj1[j]], hist=False, rug=False, ax=ax[2])
		plt.subplots_adjust(hspace=0.2)
		plt.savefig(path + '/post_dist_new.png')
		plt.close(fig)
	else:
		fig, ax = plt.subplots(2)
		for i in range(4):
			for j in range(3):
				sns.distplot(qW0_smp[:, ii0[i], jj0[j]], hist=False, rug=False, ax=ax[0])
				sns.distplot(qW1_smp[:, ii1[i], jj1[j]], hist=False, rug=False, ax=ax[1])
		plt.subplots_adjust(hspace=0.2)
		plt.savefig(path + '/post_dist_new.png')
		plt.close(fig)

if mod == 'conv_net':
	f_1 = random.sample(range(F), 2)
	f_2 = random.sample(range(F), 2)
	j_2 = random.sample(range(n_hidden), 3)
	c_1 = random.sample(range(C), 3)
	c_2 = random.sample(range(C), 3)
	i_ = random.sample(range(int(7*7*C)), 3)
	k_ = random.sample(range(K), 3)

	fig, ax = plt.subplots(2, 2)
	for f in range(2):
		for j in range(2):
			for c in range(3):
				w_samp = qW0_smp[:, f_1[f], f_2[j], 0, c_1[c]]
				sns.distplot(w_samp, hist=False, rug=False, ax=ax[0, 0])
				w_samp = qW1_smp[:, f_1[f], f_2[j], c_1[c], c_2[c]]
				sns.distplot(w_samp, hist=False, rug=False, ax=ax[0, 1])
	for i in range(3):
		for j in range(3):
			sns.distplot(qW2_smp[:, i_[i], j_2[j]], hist=False, rug=False, ax=ax[1,0])
			sns.distplot(qW3_smp[:, j_2[i], k_[j]], hist=False, rug=False, ax=ax[1,1])
	plt.subplots_adjust(hspace=0.2)
	plt.savefig(path + '/post_dist_new.png')
	plt.close(fig)