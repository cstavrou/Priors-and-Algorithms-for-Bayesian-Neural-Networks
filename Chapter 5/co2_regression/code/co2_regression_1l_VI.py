# =======================================================================================================
# Using the co2 regression dataset
# =======================================================================================================


# ============
# Description
# ============
# Performs VI - on a 1 hidden layer Bayesian NN
# Variational Mean Field (VMF) approximating distribution is used


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
from sklearn.metrics import confusion_matrix

# Load the co2 regression data
X_test = np.reshape(np.load('../data/x_test.npy'), [-1, 1])
Y_test = np.reshape(np.load('../data/y_test.npy'), [-1, 1])
X_train = np.reshape(np.load('../data/x_train.npy'), [-1, 1])
Y_train = np.reshape(np.load('../data/y_train.npy'), [-1, 1])

ed.set_seed(314159)
np.random.seed(seed=314159)
std_out = 0.07                          # output standard deviation

K = np.shape(Y_train)[1]                # number of classes.
D = np.shape(X_train)[1]

n_hidden = int(sys.argv[1])             # number of hidden units
inf_iter = int(sys.argv[2]) 		    # number of iterations of VI.
std = float(sys.argv[3])                # prior dispersion parameter (note this is then rescaled accordingly)
                                        # Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer
                                        # Laplace: b = std^2/d_j
                                        # StudentT: s = std^2/d_j
non_zero_mean = str(sys.argv[4])        # True/False (if mean of posterior distributions is 0)
# sys.argv[5] = normal/laplace/T        # prior distribution
if str(sys.argv[5]) == 'T':
	df = float(sys.argv[6])             # degrees of freedom for the T distribution

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
# Define prior graph
if str(sys.argv[5]) == 'laplace':
	W_0 = Laplace(loc=tf.zeros([D, n_hidden]), scale=std**2/D*tf.ones([D, n_hidden]))
	W_1 = Laplace(loc=tf.zeros([n_hidden, K]), scale=std**2/n_hidden*tf.ones([n_hidden, K]))
	b_0 = Laplace(loc=tf.zeros(n_hidden), scale=std**2/D*tf.ones(n_hidden))
	b_1 = Laplace(loc=tf.zeros(K), scale=std**2/n_hidden*tf.ones(K))
if str(sys.argv[5]) == 'normal':
	W_0 = Normal(loc=tf.zeros([D, n_hidden]), scale=std*D**(-.5)*tf.ones([D, n_hidden]))
	W_1 = Normal(loc=tf.zeros([n_hidden, K]), scale=std*n_hidden**(-.5)*tf.ones([n_hidden, K]))
	b_0 = Normal(loc=tf.zeros(n_hidden), scale=std*D**(-.5)*tf.ones(n_hidden))
	b_1 = Normal(loc=tf.zeros(K), scale=std*n_hidden**(-.5)*tf.ones(K))
if str(sys.argv[5]) == 'T':
	W_0 = StudentT(df=df*tf.ones([D, n_hidden]), loc=tf.zeros([D, n_hidden]), scale=std**2/D*tf.ones([D, n_hidden]))
	W_1 = StudentT(df=df*tf.ones([n_hidden, K]), loc=tf.zeros([n_hidden, K]), scale=std**2/n_hidden*tf.ones([n_hidden, K]))
	b_0 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.zeros(n_hidden), scale=std**2/D*tf.ones(n_hidden))
	b_1 = StudentT(df=df*tf.ones([K]), loc=tf.zeros(K), scale=std**2/n_hidden*tf.ones(K))
# Inputs
x = tf.placeholder(tf.float32, [None, D])
# Gaussian likelihood
y = Normal(loc=nn(x, W_0, b_0, W_1, b_1), scale=std_out*tf.ones([tf.shape(x)[0]]))
# We use a placeholder for the labels in anticipation of the traning data
y_ph = tf.placeholder(tf.float32, [None])

# Build predictive graph
#
x_pred = tf.placeholder(tf.float32, [None, None])
ww0 = tf.placeholder(tf.float32, [None, None])
ww1 = tf.placeholder(tf.float32, [None, None])
bb0 = tf.placeholder(tf.float32, [None])
bb1 = tf.placeholder(tf.float32, [None])
y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1)

# Define approximating distribution (VMF) graph
#
if non_zero_mean == 'False':
	if str(sys.argv[5]) == 'laplace':
		qW_0 = Laplace(loc=tf.Variable(tf.zeros([D, n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([D, n_hidden])))
		qW_1 = Laplace(loc=tf.Variable(tf.zeros([n_hidden, K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([n_hidden, K])))
		qb_0 = Laplace(loc=tf.Variable(tf.zeros([n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([n_hidden])))
		qb_1 = Laplace(loc=tf.Variable(tf.zeros([K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([K])))
	if str(sys.argv[5]) == 'normal':
		qW_0 = Normal(loc=tf.Variable(tf.zeros([D, n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([D, n_hidden])))
		qW_1 = Normal(loc=tf.Variable(tf.zeros([n_hidden, K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([n_hidden, K])))
		qb_0 = Normal(loc=tf.Variable(tf.zeros([n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([n_hidden])))
		qb_1 = Normal(loc=tf.Variable(tf.zeros([K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([K])))
	if str(sys.argv[5]) == 'T':
		qW_0 = StudentT(df=tf.Variable(df*tf.ones([D, n_hidden])), loc=tf.Variable(tf.zeros([D, n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([D, n_hidden])))
		qW_1 = StudentT(df=tf.Variable(df*tf.ones([n_hidden, K])), loc=tf.Variable(tf.zeros([n_hidden, K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([n_hidden, K])))
		qb_0 = StudentT(df=tf.Variable(df*tf.ones([n_hidden])), loc=tf.Variable(tf.zeros(n_hidden)), 
			scale=tf.Variable(std**2/D*tf.ones(n_hidden)))
		qb_1 = StudentT(df=tf.Variable(df*tf.ones([K])), loc=tf.Variable(tf.zeros(K)), 
			scale=tf.Variable(std**2/n_hidden*tf.ones(K)))
if non_zero_mean == 'True':
	if str(sys.argv[5]) == 'laplace':
		qW_0 = Laplace(loc=tf.Variable(tf.cast(np.random.laplace(size=[D, n_hidden], scale=std**2/D), tf.float32)), 
			scale=tf.Variable(std**2/D*tf.ones([D, n_hidden])))
		qW_1 = Laplace(loc=tf.Variable(tf.cast(np.random.laplace(size=[n_hidden, K], scale=std**2/n_hidden), tf.float32)), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([n_hidden, K])))
		qb_0 = Laplace(loc=tf.Variable(tf.cast(np.random.laplace(size=[n_hidden], scale=std**2/n_hidden), tf.float32)), 
			scale=tf.Variable(std**2/D*tf.ones([n_hidden])))
		qb_1 = Laplace(loc=tf.Variable(tf.cast(np.random.laplace(size=[K], scale=std**2/n_hidden), tf.float32)), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([K])))
	if str(sys.argv[5]) == 'normal':
		qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, n_hidden], stddev=std/D**-.5)), 
			scale=tf.Variable(std/D**-.5*tf.ones([D, n_hidden])))
		qW_1 = Normal(loc=tf.Variable(tf.random_normal([n_hidden, K], stddev=std/n_hidden**-.5)), 
			scale=tf.Variable(std/n_hidden**-.5*tf.ones([n_hidden, K])))
		qb_0 = Normal(loc=tf.Variable(tf.random_normal([n_hidden], stddev=std/D**-.5)), 
			scale=tf.Variable(std/D**-.5*tf.ones([n_hidden])))
		qb_1 = Normal(loc=tf.Variable(tf.random_normal([K], stddev=std/n_hidden**-.5)), 
			scale=tf.Variable(std/n_hidden**-.5*tf.ones([K])))
	if str(sys.argv[5]) == 'T':
		qW_0 = StudentT(df=tf.Variable(df*tf.ones([D, n_hidden])), loc=tf.Variable(tf.zeros([D, n_hidden])), 
			scale=tf.Variable(std**2/D*tf.ones([D, n_hidden])))
		qW_1 = StudentT(df=tf.Variable(df*tf.ones([n_hidden, K])), loc=tf.Variable(tf.zeros([n_hidden, K])), 
			scale=tf.Variable(std**2/n_hidden*tf.ones([n_hidden, K])))
		qb_0 = StudentT(df=tf.Variable(df*tf.ones([n_hidden])), loc=tf.Variable(tf.zeros(n_hidden)), 
			scale=tf.Variable(std**2/D*tf.ones(n_hidden)))
		qb_1 = StudentT(df=tf.Variable(df*tf.ones([K])), loc=tf.Variable(tf.zeros(K)), 
			scale=tf.Variable(std**2/n_hidden*tf.ones(K)))

# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y:y_ph})
inference.initialize(n_iter=inf_iter, n_print=100)
sess = ed.get_session()

tf.global_variables_initializer().run()

if non_zero_mean != 'True':
	name_add = 'zero_mean'
else:
	name_add = 'non_zero_mean'

# Set and create path for stored files
if str(sys.argv[5]) != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter) + 
		'rep/kl_' + name_add + '/' +  str(sys.argv[5]))
else:
	path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter) + 
		'rep/kl_' + name_add + '/T_' + str(df).replace('.','_'))

if not os.path.exists(path):
  os.makedirs(path)

# Training
test_acc = []
mini_samp = 100                           # number of samples used for the MC estimate during training

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
		# MC test MSE estimate using 100 samples
		mse_tmp = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=mini_samp)
		print('\nIter ', _+1, ' -- MSE: ', mse_tmp)
		test_acc.append(mse_tmp)		

# Save test accuracy during training
name = path + '/test_acc.csv'
np.savetxt(name, test_acc, fmt = '%.5f', delimiter=',')

## Model Evaluation
#
y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})

# MC est on test data (using 100 samples)
print("MSE on test data:")
acc1 = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=mini_samp)
print(acc1)

pred_acc_list = np.zeros([mini_samp])

pW_0, pW_1, pb_0, pb_1 = (qW_0.sample(mini_samp).eval(), qW_1.sample(mini_samp).eval(),
	qb_0.sample(mini_samp).eval(), qb_1.sample(mini_samp).eval())

for i in range(mini_samp):
	pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0[i, :, :], 
		bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :]})
	mse_tmp = mse(Y_test, pred)
	pred_acc_list[i] = mse_tmp

point_est_pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0.mean(axis=0),
	bb0: pb_0.mean(axis=0), ww1: pW_1.mean(axis=0), bb1: pb_1.mean(axis=0)})
point_est_mse = mse(Y_test, point_est_pred)

# Marginal distribution plots
#	
jj = random.sample(range(n_hidden), 9)
ii = random.sample(range(n_hidden), 9)

fig, ax = plt.subplots(2)
for i in range(9):	
	sns.distplot(qW_0.sample(500).eval()[:, 0, jj[i]], hist=False, rug=False, ax=ax[0])
	sns.distplot(qW_1.sample(500).eval()[:, ii[i], 0], hist=False, rug=False, ax=ax[1])
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.7)
plt.savefig(path + '/marginal_distribution_plots.png')
plt.close(fig)

# Final prediction using 500 samples
#
qW0_smp, qW1_smp = qW_0.sample(500).eval(), qW_1.sample(500).eval()
qb0_smp, qb1_smp = qb_0.sample(500).eval(), qb_1.sample(500).eval()

# Final prediction and plots
x_in = np.linspace(-2.5, 4, 400)
samples = np.shape(qW0_smp)[0]
y_hat_preds = np.zeros([samples, 400])
y_hat_test = np.zeros([samples, len(Y_test)])
mse_list = np.zeros([samples])


fig, ax = plt.subplots(1)

for i in range(samples):
	y_t = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: qW0_smp[i, :, :], 
		bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
	y_hat = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: qW0_smp[i, :, :], 
		bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
	y_hat_preds[i, :] = y_t
	y_hat_test[i, :] = y_hat
	ax.plot(x_in, y_t, color='pink')	
	mse_list[i] = mse(Y_test, y_hat)

# MC estimate
y_hat_ens = y_hat_preds.mean(axis=0)
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
'Test MSE (posterior) -- ' + str(acc1), 
'Point estimate MSE (mean of 500 samples) -- ' + str(point_est_mse),
'Std of MSE (100 samples from posterior) -- ' + str(np.std(pred_acc_list)),
'MC test MSE (500 samples) -- ' + str(fin_acc) + ' +/- ' + str(np.std(fin_acc)),
'Prior standard deviation -- ' + str(std)]
name = path + '/info_file.csv'
np.savetxt(name, info, fmt='%s' , delimiter=',')