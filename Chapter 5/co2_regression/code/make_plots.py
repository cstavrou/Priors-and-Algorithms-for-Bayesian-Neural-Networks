import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import timeit
import random
import cv2
import os
import seaborn as sns
import pandas as pd


# Load the prob trace data

n_hidden = int(sys.argv[1])                 # number of hidden units
method = str(sys.argv[2])                   # method used (hmc/sghmc)
prior = str(sys.argv[3])                    # prior dist (T/normal/laplace)
n_samp = int(sys.argv[4])                   # number of samples for HMC.
mod = str(sys.argv[5]) 				        # 1l/2l
plt_no = int(sys.argv[6])                   # last file number (see folder and number for last file = n_iter - 1)
trace_plt = str(sys.argv[7])                # True/False (whether to make the trace plot for the weights)
mar_plot = str(sys.argv[8])                 # True/False (whether to make the marginal dist plots)
skip_first = str(sys.argv[9])               # True/False (whether to skip the burnin iteration when plotting the accuracy curve)
samp100 = str(sys.argv[10])                 # True/False (whether to use the additional per 100 samples collected)
show_samp = str(sys.argv[11])               # True/False (whether to show all posterior samples on predictive plot)
if samp100 == 'True':
	spac_samp = int(sys.argv[12])           # Every how many samples to pick a sample
	prop_burn = float(sys.argv[13])         # Proportion of samples to burn
if str(sys.argv[3]) == 'T':
	df = float(sys.argv[14])                # degrees of freedom for the T-distribution prior

output_std = 0.07

if prior != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_samp) + 'rep/' + 
		method + '/' + prior)
else:
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_samp) + 'rep/' + 
		method + '/T_' + str(df).replace('.','_')) 

if skip_first != 'True':
	acc = pd.read_csv(path + '/test_mse.csv', header=None)

for i in range(1, plt_no+1):
	tmp = pd.read_csv(path + '/test_mse' + str(i) + '.csv', header=None)
	if skip_first != 'True' or i > 1:
		acc = np.concatenate([acc, tmp])
	else:
		acc = pd.read_csv(path + '/test_mse' + str(i) + '.csv', header=None)

fig, ax = plt.subplots(1)
ax.plot(50*np.arange(len(acc)), acc)
ax.set_xlabel('Iteration') 
ax.set_ylabel('Test MSE')
if skip_first == 'True':
	plt.savefig(path + '/mse_plot_skip_first.png')
else: 
	plt.savefig(path + '/mse_plot.png')
plt.close(fig)

if trace_plt == 'True':
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

# Load the data
X_test = np.load('../data/x_test.npy')
Y_test = np.load('../data/y_test.npy')
X_train = np.load('../data/x_train.npy')
Y_train = np.load('../data/y_train.npy')

# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.reshape(Y_test,[-1])
Y_train = np.reshape(Y_train, [-1])

if mod == '1l':
	def pred_nn(x, W_0, b_0, W_1, b_1):
	    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
	    o = tf.matmul(h, W_1) + b_1
	    return tf.reshape(o, [-1]) 
    # Build predictive graph
	x_pred = tf.placeholder(tf.float32, [None, None])
	ww0 = tf.placeholder(tf.float32, [None, None])
	ww1 = tf.placeholder(tf.float32, [None, None])
	bb0 = tf.placeholder(tf.float32, [None])
	bb1 = tf.placeholder(tf.float32, [None])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1)

if mod == '2l':
	def pred_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
	    h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
	    h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
	    o = tf.matmul(h2, W_2) + b_2
	    return tf.reshape(o, [-1])
	# Build predictive graph
	x_pred = tf.placeholder(tf.float32, [None, None])
	ww0 = tf.placeholder(tf.float32, [None, None])
	ww1 = tf.placeholder(tf.float32, [None, None])
	ww2 = tf.placeholder(tf.float32, [None, None])
	bb0 = tf.placeholder(tf.float32, [None])
	bb1 = tf.placeholder(tf.float32, [None])
	bb2 = tf.placeholder(tf.float32, [None])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2)


def mse(Y_true, Y_hat):
    sq_err = (Y_true - Y_hat)**2
    return np.mean(sq_err) 

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

# Spaces out the samples taken every 100 and removes the appropriate burnin
#
if samp100 == 'True':
	aa = np.shape(qW0_smp)[0]
	burnin = int(prop_burn*aa)
	if mod == '1l' or mod == '2l':
		qW0_smp, qW1_smp, qb0_smp, qb1_smp = (qW0_smp[burnin:aa:spac_samp, :, :], qW1_smp[burnin:aa:spac_samp, :, :],
			qb0_smp[burnin:aa:spac_samp, :], qb1_smp[burnin:aa:spac_samp, :])
	if mod == '2l':
		qW2_smp, qb2_smp = qW2_smp[burnin:aa:spac_samp, :, :], qb2_smp[burnin:aa:spac_samp, :]

# Final prediction and plots
x_in = np.linspace(-2.5, 4, 400)
samples = np.shape(qW0_smp)[0]

y_hat_preds = np.zeros([samples, 400])
y_hat_test = np.zeros([samples, len(Y_test)])
mse_list = np.zeros([samples])

with tf.Session() as sess:
	# Initialise all the vairables in the session.
	sess.run(tf.global_variables_initializer())

	fig, ax = plt.subplots(1)
	for i in range(samples):
		if mod == '1l':
			if i+1 == samples:
				point_est_y = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), 
					ww0: qW0_smp.mean(axis=0), bb0: qb0_smp.mean(axis=0), 
					ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0)})
				point_est_test = sess.run(y_pred, feed_dict={x_pred: X_test, 
					ww0: qW0_smp.mean(axis=0), bb0: qb0_smp.mean(axis=0), 
					ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0)})

			y_t = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: qW0_smp[i, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
			y_hat = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: qW0_smp[i, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
		if mod == '2l':
			if i+1 == samples:
				point_est_y = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), 
					ww0: qW0_smp.mean(axis=0), bb0: qb0_smp.mean(axis=0), ww1: qW1_smp.mean(axis=0), 
					bb1: qb1_smp.mean(axis=0), ww2: qW2_smp.mean(axis=0), bb2: qb2_smp.mean(axis=0)})
				point_est_test = sess.run(y_pred, feed_dict={x_pred: X_test, 
					ww0: qW0_smp.mean(axis=0), bb0: qb0_smp.mean(axis=0), 
					ww1: qW1_smp.mean(axis=0), bb1: qb1_smp.mean(axis=0),
					ww2: qW2_smp.mean(axis=0), bb2: qb2_smp.mean(axis=0)})
			y_t = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: qW0_smp[i, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :], 
				ww2: qW2_smp[i, :, :], bb2: qb2_smp[i, :]})
			y_hat = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: qW0_smp[i, :, :], 
				bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :],
				ww2: qW2_smp[i, :, :], bb2: qb2_smp[i, :]})
		y_hat_preds[i, :] = y_t
		y_hat_test[i, :] = y_hat	
		mse_list[i] = mse(Y_test, y_hat)
		if show_samp == 'True':
			ax.plot(x_in, y_t, color='pink')

	# MC estimate
	y_hat_ens = y_hat_preds.mean(axis=0)

	np.save(path + '/y_hat_preds.npy', y_hat_preds)
	np.save(path + '/y_hat_test.npy', y_hat_test)

	ax.plot(x_in, y_hat_ens, color='black')
	ax.plot(x_in, point_est_y, color='red') 
	ax.scatter(X_test[:, 0], Y_test)
	# Rescaling everything to find the 99% credible interval
	yerr = np.std(y_hat_preds, axis=0)*(1/output_std)*(samples)**-.5
	ax.fill_between(x_in, y_hat_ens - 2.58*yerr, y_hat_ens + 2.58*yerr, facecolor='orange', alpha=0.4)
	plt.savefig(path + '/posterior_pred_samples_new.png')
	plt.close(fig)

	fin_acc = mse(Y_test, y_hat_test.mean(axis=0))
	print('Final prediction MSE = ', fin_acc, ' +/- ', str(np.std(mse_list)))
	print('Point estimate MSE = ', mse(Y_test, point_est_test))
	print('Total number of samples collected = ', np.shape(qW0_smp)[0])

# Remake marginal model plots
jj = random.sample(range(n_hidden), 9)
ii0 = random.sample(range(n_hidden), 9)
ii1 = random.sample(range(n_hidden), 9)

if mod == '2l':
	fig, ax = plt.subplots(3)
	for i in range(9):
		sns.distplot(qW0_smp[:, 0, jj[i]], hist=False, rug=False, ax=ax[0])
		sns.distplot(qW1_smp[:, ii0[i], ii1[i]], hist=False, rug=False, ax=ax[1])
		sns.distplot(qW2_smp[:, ii1[i], 0], hist=False, rug=False, ax=ax[2])
	plt.subplots_adjust(hspace=0.2)
	plt.savefig(path + '/post_dist_final_new.png')
	plt.close(fig)

if mod == '1l':
	fig, ax = plt.subplots(2)
	for i in range(9):
		sns.distplot(qW0_smp[:, 0, jj[i]], hist=False, rug=False, ax=ax[0])
		sns.distplot(qW1_smp[:, ii0[i], 0], hist=False, rug=False, ax=ax[1])
	plt.subplots_adjust(hspace=0.2)
	plt.savefig(path + '/post_dist_final_new.png')
	plt.close(fig)