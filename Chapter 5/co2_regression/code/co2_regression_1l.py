# =======================================================================================================
# Using the co2 regression dataset
# =======================================================================================================


# ============
# Description
# ============
# Performs HMC inference on a 1 hidden layer Bayesian NN

#
# Inference performed sequentially 
# (Phase 1: burnin period) - this is later adjusted
# (Phase 2: sampling phase) 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from edward.models import Categorical, Normal, Laplace, Empirical, StudentT
import edward as ed
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
import sys
import timeit
import random
import cv2
import os

# Load the co2 regression data
X_test = np.reshape(np.load('../data/x_test.npy'), [-1, 1])
Y_test = np.reshape(np.load('../data/y_test.npy'), [-1, 1])
X_train = np.reshape(np.load('../data/x_train.npy'), [-1, 1])
Y_train = np.reshape(np.load('../data/y_train.npy'), [-1, 1])

# Old bit of code was for batch mode (batch mode has been disabled - plus use N = 2000 for
# sys.argv[1])
#
# def batch_ind(x, n_epoch, batch_size):
# 	N = np.shape(x)[0]
# 	n_batch = int(N/batch_size)
# 	indices = np.zeros([n_epoch, n_batch, batch_size]).astype(int)
# 	for i in range(n_epoch):
# 		# Shuffle indices
# 		rand = random.sample(range(N), N)
# 		for j in range(n_batch):
# 			indices[i, j, :] = rand[j*batch_size:(j+1)*batch_size]
# 	return indices

ed.set_seed(314159)
np.random.seed(seed=314159)
# N = int(sys.argv[1])   # number of images in a minibatch.
disc_fact = float(sys.argv[1])          # increase to step size with each phase iteration 
#                                         step_size_new = step_size*(1+disc_fact)^n, where n is the phase iteration
n_hidden = int(sys.argv[2])             # number of hidden units
# sys.argv[3] = hmc/sghmc               # method of inference
# sys.argv[4] = normal/laplace/T        # prior distribution
n_samp = int(sys.argv[5])               # number of samples for HMC
leap_size = float(sys.argv[6])          # step size for leapfrog scheme
step_no = int(sys.argv[7])              # number of leapfrog steps
nburn = int(sys.argv[8])                # number of units to burn every new phase 
std = float(sys.argv[9])                # prior dispersion parameter (note this is then rescaled accordingly)
                                        # Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer
                                        # Laplace: b = std^2/d_j
                                        # StudentT: s = std^2/d_j
n_iter_learn = int(sys.argv[10])        # Number of iterations of learning (i.e. number of phases)
AWS = str(sys.argv[11])                 # True/False: enables the program to run on AWS (i.e. produces no graphics just output files)
if str(sys.argv[4]) == 'T':
	df = float(sys.argv[12])            # degrees of freedom for the T distribution
std_out = 0.07                          # output standard deviation
n_examples, K = np.shape(Y_train)       # number of training examples, number of classes.
D = np.shape(X_train)[1]                # number of dimensions of X

# Reshape the labels so that they are vectors
Y_train = np.reshape(Y_train, [-1])
Y_test = np.reshape(Y_test, [-1])

# neural network used for the likelihood
def nn(x, W_0, b_0, W_1, b_1):
    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    out = tf.matmul(h, W_1) + b_1
    return tf.reshape(out, [-1])
# predictive neural network
def pred_nn(x, W_0, b_0, W_1, b_1):
    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    o = tf.matmul(h, W_1) + b_1
    return tf.reshape(o, [-1])
# mean squared error
def mse(Y_true, Y_hat):
    sq_err = (Y_true - Y_hat)**2
    return np.mean(sq_err)  

# Build predictive graph
def pred_graph():
	x_pred = tf.placeholder(tf.float32, [None, D])
	ww0 = tf.placeholder(tf.float32, [D, n_hidden])
	ww1 = tf.placeholder(tf.float32, [n_hidden, K])
	bb0 = tf.placeholder(tf.float32, [n_hidden])
	bb1 = tf.placeholder(tf.float32, [K])
	y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1)
	return x_pred, ww0, ww1, bb0, bb1, y_pred

# Build the inference graph (initial)
def ed_graph_init():
	# Graph for prior distributions
	if str(sys.argv[4]) == 'laplace':
		W_0 = Laplace(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = Laplace(loc=tf.zeros([n_hidden, K]), scale=(std**2/n_hidden)*tf.ones([n_hidden, K]))
		b_0 = Laplace(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = Laplace(loc=tf.zeros(K), scale=(std**2/n_hidden)*tf.ones(K))
	if str(sys.argv[4]) == 'normal':
		W_0 = Normal(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = Normal(loc=tf.zeros([n_hidden, K]), scale=std*n_hidden**(-.5)*tf.ones([n_hidden, K]))
		b_0 = Normal(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = Normal(loc=tf.zeros(K), scale=std*n_hidden**(-.5)*tf.ones(K))
	if str(sys.argv[4]) == 'T':
		W_0 = StudentT(df=df*tf.ones([D, n_hidden]), loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = StudentT(df=df*tf.ones([n_hidden, K]), loc=tf.zeros([n_hidden, K]), scale=std**2/n_hidden*tf.ones([n_hidden, K]))
		b_0 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = StudentT(df=df*tf.ones([K]), loc=tf.zeros(K), scale=std**2/n_hidden*tf.ones(K))
	# Inputs
	x = tf.placeholder(tf.float32, [None, D])
	# Regression likelihood
	y = Normal(loc=nn(x, W_0, b_0, W_1, b_1), scale=std_out*tf.ones([tf.shape(x)[0]]))
	# We use a placeholder for the labels in anticipation of the traning data.
	y_ph = tf.placeholder(tf.float32, [None])

	# Graph for posterior distribution
	if str(sys.argv[4]) == 'normal':
		qW_0 = Empirical(params=tf.Variable(tf.random_normal([n_samp, D, n_hidden])))
		qW_1 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden, K], stddev=std*(n_hidden**-.5))))
		qb_0 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden])))
		qb_1 = Empirical(params=tf.Variable(tf.random_normal([n_samp, K], stddev=std*(n_hidden**-.5))))
	if str(sys.argv[4]) == 'laplace' or str(sys.argv[4]) == 'T':
		# Use a placeholder otherwise cannot assign a tensor > 2GB
		w0 = tf.placeholder(tf.float32, [n_samp, D, n_hidden])
		w1 = tf.placeholder(tf.float32, [n_samp, n_hidden, K])
		b0 = tf.placeholder(tf.float32, [n_samp, n_hidden])
		b1 = tf.placeholder(tf.float32, [n_samp, K])
		# Empirical distribution
		qW_0 = Empirical(params=tf.Variable(w0))
		qW_1 = Empirical(params=tf.Variable(w1))
		qb_0 = Empirical(params=tf.Variable(b0))
		qb_1 = Empirical(params=tf.Variable(b1))
	# Build inference graph	
	if str(sys.argv[3]) == 'hmc':	
		inference = ed.HMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_ph})
	if str(sys.argv[3]) == 'sghmc':	
		inference = ed.SGHMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_ph})

	# Initialse the inference variables
	if str(sys.argv[3]) == 'hmc':
		inference.initialize(step_size = leap_size, n_steps = step_no, n_print=100)
	if str(sys.argv[3]) == 'sghmc':
		inference.initialize(step_size = leap_size, friction=0.4, n_print=100)
	
	if str(sys.argv[4]) == 'laplace' or str(sys.argv[4]) == 'T':
		return ((x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference,
			w0, w1, b0, b1)
	else:
		return (x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference		


# Inference graph (second phase)
def ed_graph_2(disc=1):
	# Priors
	if str(sys.argv[4]) == 'laplace':
		W_0 = Laplace(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = Laplace(loc=tf.zeros([n_hidden, K]), scale=(std**2/n_hidden)*tf.ones([n_hidden, K]))
		b_0 = Laplace(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = Laplace(loc=tf.zeros(K), scale=(std**2/n_hidden)*tf.ones(K))
	if str(sys.argv[4]) == 'normal':
		W_0 = Normal(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = Normal(loc=tf.zeros([n_hidden, K]), scale=std*n_hidden**(-.5)*tf.ones([n_hidden, K]))
		b_0 = Normal(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = Normal(loc=tf.zeros(K), scale=std*n_hidden**(-.5)*tf.ones(K))
	if str(sys.argv[4]) == 'T':
		W_0 = StudentT(df=df*tf.ones([D, n_hidden]), loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
		W_1 = StudentT(df=df*tf.ones([n_hidden, K]), loc=tf.zeros([n_hidden, K]), scale=std**2/n_hidden*tf.ones([n_hidden, K]))
		b_0 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
		b_1 = StudentT(df=df*tf.ones([K]), loc=tf.zeros(K), scale=std**2/n_hidden*tf.ones(K))
	# Inputs
	x = tf.placeholder(tf.float32, [None, None])
	# Regression output
	y = Normal(loc=nn(x, W_0, b_0, W_1, b_1), scale=std_out*tf.ones([tf.shape(x)[0]]))
	# We use a placeholder for the labels in anticipation of the traning data.
	y_ph = tf.placeholder(tf.float32, [None])

	# Use a placeholder for the pre-trained posteriors
	w0 = tf.placeholder(tf.float32, [n_samp, D, n_hidden])
	w1 = tf.placeholder(tf.float32, [n_samp, n_hidden, K])
	b0 = tf.placeholder(tf.float32, [n_samp, n_hidden])
	b1 = tf.placeholder(tf.float32, [n_samp, K])

	# Empirical distribution
	qW_0 = Empirical(params=tf.Variable(w0))
	qW_1 = Empirical(params=tf.Variable(w1))
	qb_0 = Empirical(params=tf.Variable(b0))
	qb_1 = Empirical(params=tf.Variable(b1))
	
	if str(sys.argv[3]) == 'hmc':	
		inference = ed.HMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_ph})
	if str(sys.argv[3]) == 'sghmc':	
		inference = ed.SGHMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_ph})

	# Initialse the inference variables
	if str(sys.argv[3]) == 'hmc':
		inference.initialize(step_size = disc*leap_size, n_steps = step_no, n_print=100)
	if str(sys.argv[3]) == 'sghmc':
		inference.initialize(step_size = disc*leap_size, friction=0.4, n_print=100)
	
	return ((x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference,
		w0, w1, b0, b1)



# ============================================================================================
# Phase 1 of learning (burnin)
# ============================================================================================

# Reset the tensorflow graph
tf.reset_default_graph()

# Build predictive graph
x_pred, ww0, ww1, bb0, bb1, y_pred = pred_graph()

# Build the initial inference graph
if str(sys.argv[4]) == 'laplace' or str(sys.argv[4]) == 'T':
	((x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, 
		qW_1, qb_1, inference, w0, w1, b0, b1) = ed_graph_init()
else:
	(x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference = ed_graph_init()

with tf.Session() as sess:

	if str(sys.argv[4]) == 'laplace' or str(sys.argv[4]) == 'T':
		# Initialise all the vairables in the session.
		init = tf.global_variables_initializer()
		if str(sys.argv[3]) != 'kl':
			if str(sys.argv[4]) == 'laplace':
				sess.run(init, feed_dict={w0: np.random.laplace(size=[n_samp, D, n_hidden]),
						w1: np.random.laplace(size=[n_samp, n_hidden, K], scale=(std**2/n_hidden)), 
						b0: np.random.laplace(size=[n_samp, n_hidden]),
						b1: np.random.laplace(size=[n_samp, K], scale=(std**2/n_hidden))})
			if str(sys.argv[4]) == 'T':
				sess.run(init, feed_dict={w0: np.random.standard_t(df, size=[n_samp, D, n_hidden]),
						w1: np.random.standard_t(df, size=[n_samp, n_hidden, K]), 
						b0: np.random.standard_t(df, size=[n_samp, n_hidden]),
						b1: np.random.standard_t(df, size=[n_samp, K])})

	if str(sys.argv[4]) == 'normal' or str(sys.argv[3]) == 'kl':
		tf.global_variables_initializer().run()

	if str(sys.argv[4]) != 'T':
		path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter*n_iter_learn) + 'rep/' + 
			str(sys.argv[3]) + '/' + str(sys.argv[4]))
	else:
		path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter*n_iter_learn) + 'rep/' + 
			str(sys.argv[3]) + '/' + 'T_' + str(df).replace('.','_'))

	if not os.path.exists(path):
	  os.makedirs(path)

	# Training - Phase 1
	test_acc = []

	for _ in range(inference.n_iter):
		# Start timer - make sure only the actual inference part is calculated
		if _ == 0:
			total = timeit.default_timer()
		start = timeit.default_timer()
		info_dict = inference.update(feed_dict={x: X_train, y_ph: Y_train})
		inference.print_progress(info_dict)
		elapsed = timeit.default_timer() - start
		total = total + elapsed
		if (_ + 1 ) % 50 == 0 or _ == 0:
			y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})
			mse_tmp = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=500)
			print('\nIter ', _+1, ' -- MSE: ', mse_tmp)
			test_acc.append(mse_tmp)		

	# Save test accuracy during training
	name = path + '/test_mse.csv'
	np.savetxt(name, test_acc, fmt = '%.5f', delimiter=',')

	## Model Evaluation
	#
	y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})
	if str(sys.argv[3]) != 'kl':
		W0_opt = (qW_0.params.eval()[nburn:, :, :]).mean(axis=0)
		W1_opt = (qW_1.params.eval()[nburn:, :, :]).mean(axis=0)
		b0_opt = (qb_0.params.eval()[nburn:, :]).mean(axis=0)
		b1_opt = (qb_1.params.eval()[nburn:, :]).mean(axis=0)

		y_post1 = ed.copy(y, {W_0: W0_opt, W_1: W1_opt, b_0: b0_opt, b_1: b1_opt})

	mini_samp = 100

	print("MSE on test data:")
	acc1 = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=500)
	print(acc1)


	print("MSE on test data: (using mean)")
	acc2 = ed.evaluate('mse', data={x: X_test, y_post1: Y_test}, n_samples=500)
	print(acc2)

	pred_mse_list = np.zeros([mini_samp])
	rnd = random.sample(range(nburn,n_samp), mini_samp)

	pW_0, pW_1, pb_0, pb_1 = (qW_0.params.eval()[rnd, :, :], qW_1.params.eval()[rnd, :, :], 
		qb_0.params.eval()[rnd, :], qb_1.params.eval()[rnd, :])

	for i in range(mini_samp):
		pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0[i, :, :], 
			bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :]})
		mse_tmp = mse(Y_test, pred)
		pred_mse_list[i] = mse_tmp

	point_est_pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0.mean(axis=0),
		bb0: pb_0.mean(axis=0), ww1: pW_1.mean(axis=0), bb1: pb_1.mean(axis=0)})
	point_est_mse = mse(Y_test, point_est_pred)

	print('Point estimate MSE -- ', point_est_mse)

	# Delete unnecessary variables to free up memory
	del pW_0, pW_1, pb_0, pb_1

	# Trace plot
	#
	n_lags_used = n_samp - nburn
	acf_vals = np.zeros([18, n_lags_used])
	rnd0 = random.sample(range(n_hidden), 9)
	rnd1 = random.sample(range(n_hidden), 9)
	trace_plotw = np.zeros([9, n_samp])

	for i in range(9):
		w_samp = qW_0.params.eval()[:, 0, rnd0[i]]
		acf_vals[i,:] = acf(w_samp[nburn:], nlags=n_lags_used)
		trace_plotw[i, :] = w_samp
	np.save(path + '/traceplot_w0.npy', np.reshape(trace_plotw, [-1, 9, n_samp]))

	for i in range(9):
		w_samp = qW_1.params.eval()[:, rnd1[i], 0]
		acf_vals[9+i, :] = acf(w_samp[nburn:], nlags=n_lags_used)
		trace_plotw[i, :] = w_samp
	np.save(path + '/traceplot_w1.npy', np.reshape(trace_plotw, [-1, 9, n_samp]))

	# Auto-correlations to find the effective sample size
	#
	n_vec = np.zeros(18)

	for i in range(18):
		j = 0
		while acf_vals[i,j] > 0:
			n_vec[i] = j
			j = j + 1
	n_vec = n_vec.astype(np.int32)
	auto_corr_sum = 0

	for i in range(18):
		auto_corr_sum = auto_corr_sum + np.sum(acf_vals[i, 0:n_vec[i]])
	n_eff = int(1 + auto_corr_sum/9)

	print('N_eff = ', n_eff)

	W0_learnt = qW_0.sample(n_samp).eval()
	W1_learnt = qW_1.sample(n_samp).eval()
	b0_learnt = qb_0.sample(n_samp).eval()
	b1_learnt = qb_1.sample(n_samp).eval()

# =============================================================================================================
# Phase 2 of the training
# =============================================================================================================

inc_step = 1.0

for phase_iter in range(1, n_iter_learn):
	# Reset the tensorflow graph - and delete unnecessary variables
	del x_pred, ww0, ww1, bb0, bb1, y_pred
	if phase_iter == 1 and str(sys.argv[4]) == 'normal':
		del (x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference
	else:
		del ((x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, qW_1, qb_1, inference, w0, w1, b0, b1)
	del y_post, y_post1

	# Increase the step size gradually for the first 2/3 of learning phases, then decrease for the last 1/3
	if phase_iter/n_iter_learn < 2/3:
		inc_step = (1.0 + disc_fact)*inc_step
	else:
		inc_step = (1.0 - disc_fact)*inc_step

	print('Iteration = ', str(phase_iter), ' -- Inc_step = ', str(inc_step))
	print('\nTotal elapsed time = ', total)

	tf.reset_default_graph()

	# Build predictive graph
	x_pred, ww0, ww1, bb0, bb1, y_pred = pred_graph()

	# Build the initial graph inference graph
	((x, y), y_ph, W_0, b_0, W_1, b_1, qW_0, qb_0, 
		qW_1, qb_1, inference, w0, w1, b0, b1) = ed_graph_2(inc_step)

	with tf.Session() as sess:

		# Initialise all the vairables in the session.
		init = tf.global_variables_initializer()

		sess.run(init, feed_dict={w0: W0_learnt, w1: W1_learnt, b0: b0_learnt, b1: b1_learnt})

		del W0_learnt, W1_learnt, b0_learnt, b1_learnt

		# Training 
		test_acc = []

		for _ in range(inference.n_iter):
			start = timeit.default_timer()
			info_dict = inference.update(feed_dict={x: X_train, y_ph: Y_train})
			inference.print_progress(info_dict)
			elapsed = timeit.default_timer() - start
			total = total + elapsed
			if (_ + 1 ) % 50 == 0 or _ == 0:
				y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})
				mse_tmp = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=500)
				print('\nIter ', _+1, ' -- MSE: ', mse_tmp)
				test_acc.append(mse_tmp)		

		# Save test accuracy during training
		name = path + '/test_mse' + str(phase_iter) + '.csv'
		np.savetxt(name, test_acc, fmt = '%.5f', delimiter=',')

		## Model Evaluation
		#
		y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})

		W0_opt = (qW_0.params.eval()[nburn:, :, :]).mean(axis=0)
		W1_opt = (qW_1.params.eval()[nburn:, :, :]).mean(axis=0)
		b0_opt = (qb_0.params.eval()[nburn:, :]).mean(axis=0)
		b1_opt = (qb_1.params.eval()[nburn:, :]).mean(axis=0)

		y_post1 = ed.copy(y, {W_0: W0_opt, W_1: W1_opt, b_0: b0_opt, b_1: b1_opt})

		mini_samp = 500

		print("MSE on test data:")
		acc1 = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=500)
		print(acc1)

		print("MSE on test data: (using mean)")
		acc2 = ed.evaluate('mse', data={x: X_test, y_post1: Y_test}, n_samples=500)
		print(acc2)

		pred_mse_list = np.zeros([mini_samp])
		rnd = random.sample(range(nburn,n_samp), mini_samp)

		pW_0, pW_1, pb_0, pb_1 = (qW_0.params.eval()[rnd, :, :], qW_1.params.eval()[rnd, :, :], 
			qb_0.params.eval()[rnd, :], qb_1.params.eval()[rnd, :])

		for i in range(mini_samp):
			pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0[i, :, :], 
				bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :]})
			mse_tmp = mse(Y_test, pred)
			pred_mse_list[i] = mse_tmp

		point_est_pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0.mean(axis=0), 
			bb0: pb_0.mean(axis=0), ww1: pW_1.mean(axis=0), bb1: pb_1.mean(axis=0)})
		point_est_mse = mse(Y_test, point_est_pred)

		print('Point estimate (sample mean) accuracy -- ', point_est_mse)

		del pW_0, pW_1, pb_0, pb_1

		for i in range(9):
			w_samp = qW_0.params.eval()[:, 0, rnd0[i]]
			acf_vals[i,:] = acf(w_samp[nburn:], nlags=n_lags_used)
			trace_plotw[i, :] = w_samp
		tmp_tr = np.load(path + '/traceplot_w0.npy')
		np.save(path + '/traceplot_w0.npy', 
			np.concatenate([tmp_tr, np.reshape(trace_plotw, [-1, 9, n_samp])], 0))

		for i in range(9):
			w_samp = qW_1.params.eval()[:, rnd1[i], 0]
			acf_vals[9+i, :] = acf(w_samp[nburn:], nlags=n_lags_used)
			trace_plotw[i, :] = w_samp
		tmp_tr = np.load(path + '/traceplot_w1.npy')
		np.save(path + '/traceplot_w1.npy', 
			np.concatenate([tmp_tr, np.reshape(trace_plotw, [-1, 9, n_samp])], 0))
		# Auto-correlations to find the effective sample size
		#
		n_vec = np.zeros(18)

		for i in range(18):
			j = 0
			while acf_vals[i,j] > 0:
				n_vec[i] = j
				j = j + 1
		n_vec = n_vec.astype(np.int32)
		auto_corr_sum = 0

		for i in range(18):
			auto_corr_sum = auto_corr_sum + np.sum(acf_vals[i, 0:n_vec[i]])
		n_eff = int(1 + auto_corr_sum/9)

		print('N_eff = ', n_eff)

		if phase_iter == 1:
			# Collected samples
			tmp = qW_0.params.eval()[nburn:n_samp:n_eff, :, :]
			np.save(path + '/qW0_samp.npy', tmp)
			tmp = qW_1.params.eval()[nburn:n_samp:n_eff, :, :]
			np.save(path + '/qW1_samp.npy', tmp)
			tmp = qb_0.params.eval()[nburn:n_samp:n_eff, :]
			np.save(path + '/qb0_samp.npy', tmp)
			tmp = qb_1.params.eval()[nburn:n_samp:n_eff, :]
			np.save(path + '/qb1_samp.npy', tmp)

			tmp = qW_0.params.eval()[nburn:n_samp:100, :, :]
			np.save(path + '/qW0_samp_n100.npy', tmp)
			tmp = qW_1.params.eval()[nburn:n_samp:100, :, :]
			np.save(path + '/qW1_samp_n100.npy', tmp)
			tmp = qb_0.params.eval()[nburn:n_samp:100, :]
			np.save(path + '/qb0_samp_n100.npy', tmp)
			tmp = qb_1.params.eval()[nburn:n_samp:100, :]
			np.save(path + '/qb1_samp_n100.npy', tmp)

			# Plot marginal distribution plots
			# (plotting only a sample of the plots)
			jj = random.sample(range(n_hidden), 9)
			ii = random.sample(range(n_hidden), 9)
		else:
			# Collected samples
			tmp = np.load(path + '/qW0_samp.npy')
			np.save(path + '/qW0_samp.npy', 
				np.concatenate([tmp, qW_0.params.eval()[nburn:n_samp:n_eff, :, :]], 0))
			tmp = np.load(path + '/qW1_samp.npy')  
			np.save(path + '/qW1_samp.npy', 
				np.concatenate([tmp, qW_1.params.eval()[nburn:n_samp:n_eff, :, :]], 0))
			tmp = np.load(path + '/qb0_samp.npy')
			np.save(path + '/qb0_samp.npy', 
				np.concatenate([tmp, qb_0.params.eval()[nburn:n_samp:n_eff, :]], 0))
			tmp = np.load(path + '/qb1_samp.npy')
			np.save(path + '/qb1_samp.npy', 
				np.concatenate([tmp, qb_1.params.eval()[nburn:n_samp:n_eff, :]], 0))

			tmp = np.load(path + '/qW0_samp_n100.npy')
			np.save(path + '/qW0_samp_n100.npy', 
				np.concatenate([tmp, qW_0.params.eval()[nburn:n_samp:100, :, :]], 0))
			tmp = np.load(path + '/qW1_samp_n100.npy')  
			np.save(path + '/qW1_samp_n100.npy', 
				np.concatenate([tmp, qW_1.params.eval()[nburn:n_samp:100, :, :]], 0))
			tmp = np.load(path + '/qb0_samp_n100.npy')
			np.save(path + '/qb0_samp_n100.npy', 
				np.concatenate([tmp, qb_0.params.eval()[nburn:n_samp:100, :]], 0))
			tmp = np.load(path + '/qb1_samp_n100.npy')
			np.save(path + '/qb1_samp_n100.npy', 
				np.concatenate([tmp, qb_1.params.eval()[nburn:n_samp:100, :]], 0))  

		# Sample from current estimate for the posterior
		#
		W0_learnt = qW_0.sample(n_samp).eval()
		W1_learnt = qW_1.sample(n_samp).eval()
		b0_learnt = qb_0.sample(n_samp).eval()
		b1_learnt = qb_1.sample(n_samp).eval()

# Save collected samples
#
qW0_smp = np.load(path + '/qW0_samp.npy')
qW1_smp = np.load(path + '/qW1_samp.npy')
qb0_smp = np.load(path + '/qb0_samp.npy')
qb1_smp = np.load(path + '/qb1_samp.npy')

print('Total number of samples collected -- ', np.shape(qW0_smp)[0])

# Plot the final marginal distribution plots
# (plotting only a sample of the plots)

if AWS == 'False':
	fig, ax = plt.subplots(2)
	for i in range(9):
		sns.distplot(qW0_smp[:, 0, jj[i]], hist=False, rug=False, ax=ax[0])
		sns.distplot(qW1_smp[:, ii[i], 0], hist=False, rug=False, ax=ax[1])
	plt.subplots_adjust(hspace=0.2)
	plt.savefig(path + '/post_dist_final.png')
	plt.close(fig)

# Final prediction and plots
x_in = np.linspace(-2.5, 4, 400)
samples = np.shape(qW0_smp)[0]
y_hat_preds = np.zeros([samples, 400])
y_hat_test = np.zeros([samples, len(Y_test)])
mse_list = np.zeros([samples])

tf.reset_default_graph()
# Build predictive graph
x_pred, ww0, ww1, bb0, bb1, y_pred = pred_graph()

with tf.Session() as sess:
	# Initialise all the vairables in the session.
	sess.run(tf.global_variables_initializer())

	if AWS == 'False':
		fig, ax = plt.subplots(1)

	for i in range(samples):
		y_t = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: qW0_smp[i, :, :], 
			bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
		y_hat = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: qW0_smp[i, :, :], 
			bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
		y_hat_preds[i, :] = y_t
		y_hat_test[i, :] = y_hat
		if AWS == 'False':
			ax.plot(x_in, y_t, color='pink')	
		mse_list[i] = mse(Y_test, y_hat)

	# MC estimate
	y_hat_ens = y_hat_preds.mean(axis=0)
	if AWS == 'False':
		ax.plot(x_in, y_hat_ens, color='black') 
		ax.scatter(X_test[:, 0], Y_test)
		# Rescaling everything to find the 99% credible interval
		yerr = np.std(y_hat_preds, axis=0)*(1/std_out)*(samples)**-.5
		ax.fill_between(x_in, y_hat_ens - 2.58*yerr, y_hat_ens + 2.58*yerr, facecolor='orange', alpha=0.4)
		plt.savefig(path + '/posterior_pred_samples.png')
		plt.close(fig)

	fin_acc = mse(Y_test, y_hat_test.mean(axis=0))
	print('Final prediction accuracy = ', fin_acc, ' +/- ', str(np.std(mse_list)))

	# Save info file
	print('Total time elapsed (seconds): ',total)
	info = ['Total algorithm time (seconds) -- ' + str(total), 
	'Test MSE (last training iteration) -- ' + str(acc1), 
	'Point estimate test MSE -- ' + str(point_est_mse),
	'Std of prediction MSE (500 samples from posterior) -- ' + str(np.std(pred_mse_list)),
	'Final MC MSE -- ' + str(fin_acc) + ' +/- ' + str(np.std(mse_list)),
	'Effective Sample n -- ' + str(n_eff), 'Prior standard deviation -- ' + str(std),
	'Total number of samples collected -- ' + str(samples), 'Discount factor -- ' + str(1.0 + disc_fact)]
	if str(sys.argv[3]) == 'hmc': 
		info.append('Point estimate test MSE (during training) -- ' + str(acc2))
		info.append('Leapfrog step size -- ' + str(leap_size))
		info.append('Number of leapfrog steps -- ' + str(step_no))
		info.append('Burnin --' + str(nburn))
	if str(sys.argv[3]) == 'sghmc':
		info.append('Point estimate test MSE (during training) -- ' + str(acc2))
		info.append('Leapfrog step size -- ' + str(leap_size))
		info.append('Burnin --' + str(nburn))
	name = path + '/info_file.csv'
	np.savetxt(name, info, fmt='%s' , delimiter=',')