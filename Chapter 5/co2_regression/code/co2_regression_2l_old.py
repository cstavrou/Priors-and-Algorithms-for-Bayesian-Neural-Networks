## Using the co2 regression data

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal, Laplace, Empirical, StudentT, StudentTWithAbsDfSoftplusScale
import edward as ed
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pandas as pd
import sys
import timeit
import random
import cv2
import os


# Load the co2 regression data
X_test = np.load('../data/x_test.npy')
X_train = np.load('../data/x_train.npy')
Y_test = np.load('../data/y_test.npy')
Y_train = np.load('../data/y_train.npy')


ed.set_seed(314159)
np.random.seed(seed=314159)

n_hidden = int(sys.argv[1])           # number of hidden units.
N , D = np.shape(X_train)             # number of features.
K = np.shape(Y_test)[1]               # number of classes.
n_samp = int(sys.argv[4])             # number of samples for HMC.
inf_iter = int(sys.argv[5]) 		  # number of iterations of VI.
leap_size = float(sys.argv[6])        # leapfrog step size.
step_no = int(sys.argv[7])            # number of leapfrog steps.
nburn = int(sys.argv[8])              # number of burned samples.
std = float(sys.argv[9])              # prior standard deviation.
if str(sys.argv[3]) == 'T':           # only applies to T distribution.
	df = float(sys.argv[10])          # number of degrees of freedom.
std_out = 0.07                        # output standard deviation.

# Reshape the labels so that they are vectors
Y_train = np.reshape(Y_train, [-1])
Y_test = np.reshape(Y_test, [-1])

def nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
    h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
    out = tf.matmul(h2, W_2) + b_2
    return tf.reshape(out, [-1])

def pred_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
    h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
    o = tf.matmul(h2, W_2) + b_2
    return tf.reshape(o, [-1])

def mse(Y_true, Y_hat):
    sq_err = (Y_true - Y_hat)**2
    return np.mean(sq_err)  


if str(sys.argv[3]) == 'laplace':
	W_0 = Laplace(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
	W_1 = Laplace(loc=tf.zeros([n_hidden, n_hidden]), scale=std**2*(n_hidden**-1)*tf.ones([n_hidden, n_hidden]))
	W_2 = Laplace(loc=tf.zeros([n_hidden, K]), scale=std**2*(n_hidden**-1)*tf.ones([n_hidden, K]))
	b_0 = Laplace(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
	b_1 = Laplace(loc=tf.zeros(n_hidden), scale=std**2*(n_hidden**-1)*tf.ones(n_hidden))
	b_2 = Laplace(loc=tf.zeros(K), scale=std**2*(n_hidden**-1)*tf.ones(K))

if str(sys.argv[3]) == 'normal':
	W_0 = Normal(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
	W_1 = Normal(loc=tf.zeros([n_hidden, n_hidden]), scale=std*(n_hidden**-.5)*tf.ones([n_hidden, n_hidden]))
	W_2 = Normal(loc=tf.zeros([n_hidden, K]), scale=std*(n_hidden**-.5)*tf.ones([n_hidden, K]))
	b_0 = Normal(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
	b_1 = Normal(loc=tf.zeros(n_hidden), scale=std*(n_hidden**-.5)*tf.ones(n_hidden))
	b_2 = Normal(loc=tf.zeros(K), scale=std*(n_hidden**-.5)*tf.ones(K))

if str(sys.argv[3]) == 'T':
	W_0 = StudentT(df=df*tf.ones([D, n_hidden]), loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]))
	W_1 = StudentT(df=df*tf.ones([n_hidden, n_hidden]), loc=tf.zeros([n_hidden, n_hidden]), scale=abs(df/(df-2))*n_hidden**(-.5)*tf.ones([n_hidden, n_hidden]))
	W_2 = StudentT(df=df*tf.ones([n_hidden, K]), loc=tf.zeros([n_hidden, K]), scale=abs(df/(df-2))*n_hidden**(-.5)*tf.ones([n_hidden, K]))
	b_0 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden))
	b_1 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.zeros(n_hidden), scale=abs(df/(df-2))*n_hidden**(-.5)*tf.ones(n_hidden))
	b_2 = StudentT(df=df*tf.ones([K]), loc=tf.zeros(K), scale=abs(df/(df-2))*n_hidden**(-.5)*tf.ones(K))
	
x = tf.placeholder(tf.float32, [None, None])
y = Normal(loc=nn(x, W_0, b_0, W_1, b_1, W_2, b_2), scale=std_out*tf.ones([tf.shape(x)[0]]))
# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.float32, [N])


# Build predictive graph
#
x_pred = tf.placeholder(tf.float32, [None, D])
ww0 = tf.placeholder(tf.float32, [D, n_hidden])
ww1 = tf.placeholder(tf.float32, [n_hidden, n_hidden])
ww2 = tf.placeholder(tf.float32, [n_hidden, K])
bb0 = tf.placeholder(tf.float32, [n_hidden])
bb1 = tf.placeholder(tf.float32, [n_hidden])
bb2 = tf.placeholder(tf.float32, [K])
y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1, ww2, bb2)

if str(sys.argv[2]) == 'kl':
	if str(sys.argv[3]) == 'laplace':
		qW_0 = Laplace(loc=tf.Variable(tf.random_normal([D, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, n_hidden]))))
		qW_1 = Laplace(loc=tf.Variable(tf.random_normal([n_hidden, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_hidden]))))
		qW_2 = Laplace(loc=tf.Variable(tf.random_normal([n_hidden, K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, K]))))
		qb_0 = Laplace(loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_1 = Laplace(loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_2 = Laplace(loc=tf.Variable(tf.random_normal([K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
	if str(sys.argv[3]) == 'normal':
		qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, n_hidden]))))
		qW_1 = Normal(loc=tf.Variable(tf.random_normal([n_hidden, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_hidden]))))
		qW_2 = Normal(loc=tf.Variable(tf.random_normal([n_hidden, K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, K]))))
		qb_0 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_1 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_2 = Normal(loc=tf.Variable(tf.random_normal([K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
	# Note the scale and loc parameters here are used so that the T distribution 
	# variance is defined for df < 2
	if str(sys.argv[3]) == 'T':
		qW_0 = StudentT(df=df*tf.ones([D, n_hidden]), loc=tf.Variable(tf.random_normal([D, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, n_hidden]))))
		qW_1 = StudentT(df=df*tf.ones([n_hidden, n_hidden]), loc=tf.Variable(tf.random_normal([n_hidden, n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_hidden]))))
		qW_2 = StudentT(df=df*tf.ones([n_hidden, K]), loc=tf.Variable(tf.random_normal([n_hidden, K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, K]))))
		qb_0 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_1 = StudentT(df=df*tf.ones([n_hidden]), loc=tf.Variable(tf.random_normal([n_hidden])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
		qb_2 = StudentT(df=df*tf.ones([K]), loc=tf.Variable(tf.random_normal([K])), 
			scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))


# Define the VI inference technique, ie. minimise the KL divergence between q and p.
if str(sys.argv[2]) == 'kl':
	inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2}, data={y:y_ph})

if str(sys.argv[2]) != 'kl':
	if str(sys.argv[3]) == 'normal':
		qW_0 = Empirical(params=tf.Variable(tf.random_normal([n_samp, D, n_hidden])))
		qW_1 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden, n_hidden], stddev=std*(n_hidden**-.5))))
		qW_2 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden, K], stddev=std*(n_hidden**-.5))))
		qb_0 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden])))
		qb_1 = Empirical(params=tf.Variable(tf.random_normal([n_samp, n_hidden], stddev=std*(n_hidden**-.5))))
		qb_2 = Empirical(params=tf.Variable(tf.random_normal([n_samp, K], stddev=std*(n_hidden**-.5))))

	if str(sys.argv[3]) == 'laplace' or str(sys.argv[3]) == 'T':
		# Use a placeholder otherwise cannot assign a tensor > 2GB
		p0 = tf.placeholder(tf.float32, [n_samp, D, n_hidden])
		p1 = tf.placeholder(tf.float32, [n_samp, n_hidden, n_hidden])
		p2 = tf.placeholder(tf.float32, [n_samp, n_hidden, K])
		pp0 = tf.placeholder(tf.float32, [n_samp, n_hidden])
		pp1 = tf.placeholder(tf.float32, [n_samp, n_hidden])
		pp2 = tf.placeholder(tf.float32, [n_samp, K])

		w0 = tf.Variable(p0)
		w1 = tf.Variable(p1)
		w2 = tf.Variable(p2)		
		b0 = tf.Variable(pp0)		
		b1 = tf.Variable(pp1)
		b2 = tf.Variable(pp2)
		# Empirical distribution
		qW_0 = Empirical(params=w0)
		qW_1 = Empirical(params=w1)
		qW_2 = Empirical(params=w2)
		qb_0 = Empirical(params=b0)
		qb_1 = Empirical(params=b1)
		qb_2 = Empirical(params=b2)
	
	if str(sys.argv[2]) == 'hmc':	
		inference = ed.HMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2}, data={y: y_ph})
	if str(sys.argv[2]) == 'sghmc':	
		inference = ed.SGHMC({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2}, data={y: y_ph})


# Initialse the infernce variables
if str(sys.argv[2]) == 'hmc':
	inference.initialize(step_size = leap_size, n_steps = step_no, n_print=100)
if str(sys.argv[2]) == 'sghmc':
	inference.initialize(step_size = leap_size, friction=0.4, n_print=100)
if str(sys.argv[2]) == 'kl':
	inference.initialize(n_iter=inf_iter, n_print=100)

# sess = tf.InteractiveSession()
sess = ed.get_session()

if str(sys.argv[3]) == 'laplace' or str(sys.argv[3]) == 'T':
	# Initialise all the vairables in the session.
	init = tf.global_variables_initializer()
	if str(sys.argv[2]) != 'kl':
		if str(sys.argv[3]) == 'laplace':
			sess.run(init, feed_dict={p0: np.random.laplace(size=[n_samp, D, n_hidden]),
					p1: np.random.laplace(size=[n_samp, n_hidden, n_hidden], scale = std**2*(n_hidden**-1)),
					p2: np.random.laplace(size=[n_samp, n_hidden, K], scale = std**2*(n_hidden**-1)), 
					pp0: np.random.laplace(size=[n_samp, n_hidden]),
					pp1: np.random.laplace(size=[n_samp, n_hidden], scale = std**2*(n_hidden**-1)),
					pp2: np.random.laplace(size=[n_samp, K], scale = std**2*(n_hidden**-1))})
		if str(sys.argv[3]) == 'T':
			sess.run(init, feed_dict={p0: np.random.standard_t(df, size=[n_samp, D, n_hidden]),
					p1: np.random.standard_t(df, size=[n_samp, n_hidden, n_hidden]),
					p2: np.random.standard_t(df, size=[n_samp, n_hidden, K]),
					pp0: np.random.standard_t(df, size=[n_samp, n_hidden]),
					pp1: np.random.standard_t(df, size=[n_samp, n_hidden]),
					pp2: np.random.standard_t(df, size=[n_samp, K])})

if str(sys.argv[3]) == 'normal' or str(sys.argv[2]) == 'kl':
	tf.global_variables_initializer().run()

if str(sys.argv[3]) != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/2l_' + str(inference.n_iter) + 'rep/' + 
		str(sys.argv[2]) + '/' + str(sys.argv[3]))
else:
	path =  ('../saved/' + str(n_hidden) +'units/2l_' + str(inference.n_iter) + 'rep/' + 
		str(sys.argv[2]) + '/' + 'T_' + str(df).replace('.',''))

if not os.path.exists(path):
  os.makedirs(path)

# Training
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
	if (_ + 1) % 50 == 0 or _ == 0:
		y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, W_2: qW_2, b_0: qb_0, b_1: qb_1, b_2: qb_2})
		acc_tmp = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=100)
		print('\nIter ', _+1, ' -- MSE: ', acc_tmp)
		test_acc.append(acc_tmp)		

# Save test accuracy during training
name = path + '/test_mse.csv'
np.savetxt(name, test_acc, fmt = '%.5f', delimiter=',')


## Model Evaluation
#
y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, W_2: qW_2, b_0: qb_0, b_1: qb_1, b_2: qb_2})
if str(sys.argv[2]) != 'kl':
	W0_opt = (qW_0.params.eval()[nburn:, :, :]).mean(axis=0)
	W1_opt = (qW_1.params.eval()[nburn:, :, :]).mean(axis=0)
	W2_opt = (qW_2.params.eval()[nburn:, :, :]).mean(axis=0)
	b0_opt = (qb_0.params.eval()[nburn:, :]).mean(axis=0)
	b1_opt = (qb_1.params.eval()[nburn:, :]).mean(axis=0)
	b2_opt = (qb_2.params.eval()[nburn:, :]).mean(axis=0)

	y_post1 = ed.copy(y, {W_0: W0_opt, W_1: W1_opt, W_2: W2_opt, b_0: b0_opt, 
		b_1: b1_opt, b_2: b2_opt})


mini_samp = 100

print("MSE on test data:")
acc1 = ed.evaluate('mse', data={x: X_test, y_post: Y_test}, n_samples=100)
print(acc1)

if str(sys.argv[2]) != 'kl':
	print("MSE on test data: (using mean)")
	acc2 = ed.evaluate('mse', data={x: X_test, y_post1: Y_test}, n_samples=100)
	# acc2 = ed.evaluate('sparse_categorical_accuracy', data={x: X_test, y_post1: Y_test})  
	print(acc2)

	mse_list = np.zeros([mini_samp])
	preds = np.zeros([mini_samp, len(Y_test)])

	rnd = random.sample(range(len(range(nburn,n_samp))), mini_samp)
	pW_0, pW_1, pW_2, pb_0, pb_1, pb_2 = (qW_0.params.eval()[rnd, :, :], 
		qW_1.params.eval()[rnd, :, :],
		qW_2.params.eval()[rnd, :, :],
		qb_0.params.eval()[rnd, :],
		qb_1.params.eval()[rnd, :],
		qb_2.params.eval()[rnd, :])

	for i in range(mini_samp):
		pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0[i, :, :], 
			bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :], 
			ww2: pW_2[i, :, :], bb2: pb_2[i, :]})
		preds[i, :] = pred
		mse_tmp = mse(Y_test, pred)
		mse_list[i] = mse_tmp

	file_name = path + '/predictions_samples_not_burnin.npy'
	np.save(file_name, preds)
	file_name = path + '/mse_samples_not_burnin.npy'
	np.save(file_name, mse_list)

if str(sys.argv[2]) == 'kl':

	W0_opt = (qW_0.params.eval()[nburn:, :, :]).mean(axis=0)
	W1_opt = (qW_1.params.eval()[nburn:, :, :]).mean(axis=0)
	W2_opt = (qW_2.params.eval()[nburn:, :, :]).mean(axis=0)
	b0_opt = (qb_0.params.eval()[nburn:, :]).mean(axis=0)
	b1_opt = (qb_1.params.eval()[nburn:, :]).mean(axis=0)
	b2_opt = (qb_2.params.eval()[nburn:, :]).mean(axis=0)

	mse_list = np.zeros([mini_samp])
	preds = np.zeros([mini_samp, len(Y_test)])

	pW_0, pW_1, pW_2, pb_0, pb_1, pb_2 = (qW_0.sample(mini_samp).eval(), qW_1.sample(mini_samp).eval(),
		qW_2.sample(mini_samp).eval(), qb_0.sample(mini_samp).eval(), 
		qb_1.sample(mini_samp).eval(), qb_2.sample(mini_samp).eval())

	for i in range(mini_samp):
		pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0[i, :, :], 
			bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :], 
			ww2: pW_2[i, :, :], bb2: pb_2[i, :]})
		preds[i, :] = pred
		mse_tmp = mse(Y_test, pred)
		mse_list[i] = mse_tmp

	file_name = path + '/mse_' + str(mini_samp) + 'samples_from_posterior.npy'
	np.save(file_name, mse_list)

	file_name = path + '/predictions_' + str(mini_samp) + 'samples_from_posterior.npy'
	np.save(file_name, preds)

mc_pred = sess.run(y_pred, feed_dict={x_pred: X_test, ww0: pW_0.mean(axis=0),
	bb0: pb_0.mean(axis=0), ww1: pW_1.mean(axis=0), bb1: pb_1.mean(axis=0),
	ww2: pW_2.mean(axis=0), bb2: pb_2.mean(axis=0)})
mc_mse = mse(Y_test, mc_pred)

# Save info file
print('Total time elapsed (seconds): ',total)
info = ['Total algorithm time (seconds) -- ' + str(total), 'Batch size -- ' + str(N), 
'Test accuracy (posterior) -- ' + str(acc1), 
'MSE (500 samples from posterior MC estimate) -- ' + str(mc_mse),
'Std of MSE estimate (' + str(mini_samp) + ' samples from posterior) -- ' + str(np.std(mse_list)),
'Prior standard deviation -- ' + str(std),
'Output noise (std) -- ' + str(std_out)]
if str(sys.argv[2]) == 'hmc': 
	info.append('Test accuracy (' + str(mini_samp) + ' sample MSE estimate exluding burn-in) -- ' + str(acc2))
	info.append('Leapfrog step size -- ' + str(leap_size))
	info.append('Number of leapfrog steps -- ' + str(step_no))
	info.append('Burnin --' + str(nburn))
if str(sys.argv[2]) == 'sghmc':
	info.append('Test accuracy (mean of samples exluding burnin) -- ' + str(acc2))
	info.append('Leapfrog step size -- ' + str(leap_size))
	info.append('Burnin --' + str(nburn))
name = path + '/info_file.csv'
np.savetxt(name, info, fmt='%s' , delimiter=',')


# Plot marginal distribution plots

if str(sys.argv[2]) == 'hmc' or str(sys.argv[2]) == 'sghmc':
	ii = random.sample(range(n_hidden), 10)
	kk = random.sample(range(K), 3)
	jj = random.sample(range(n_hidden), 3)
	for i in range(10):
		sns.distplot(qW_0.params.eval()[nburn:, 0, ii[i]], hist=False, rug=False)
	plt.show()
	for i in range(10):
		sns.distplot(qW_1.params.eval()[nburn:, 0, ii[i]], hist=False, rug=False)
	plt.show()
	for k in range(3):
		for j in range(3):
			sns.distplot(qW_2.params.eval()[nburn:, jj[j], kk[k]], hist=False, rug=False)
	plt.show()
	# Partial auto-correlation plot
	for i in range(10):
		series = pd.Series(qW_0.params.eval()[:, 0, ii[i]])
		plot_pacf(series, lags=100)
		plt.show()
	for i in range(10):
		series = pd.Series(qW_0.params.eval()[:, 0, ii[i]])
		plot_acf(series, lags=400)
		plt.show()
	# for i in range(10):
	# 	series = pd.Series(qW_1.params.eval()[:, 0, ii[i]])
	# 	plot_pacf(series, lags=100)
	# 	plt.show()
	# for i in range(10):
	# 	series = pd.Series(qW_1.params.eval()[:, 0, ii[i]])
	# 	plot_acf(series, lags=400)
	# 	plt.show()

if str(sys.argv[2]) == 'kl':
	for i in range(10):
		sns.distplot(qW_0.sample(10).eval()[:, 0, i], hist=False, rug=False)
	plt.show()
	for i in range(10):
		sns.distplot(qW_1.sample(10).eval()[:, 0, i], hist=False, rug=False)
	plt.show()
	for i in range(10):
		sns.distplot(qW_2.sample(10).eval()[:, i, 0], hist=False, rug=False)
	plt.show()

# Graph of drawn functions from the posterior
x_in = np.linspace(-2.5, 4, 400)
samples = 30
W0_samp, W1_samp, W2_samp, b0_samp, b1_samp, b2_samp = (qW_0.sample(samples).eval(), 
	qW_1.sample(samples).eval(), qW_2.sample(samples).eval(), qb_0.sample(samples).eval(), 
	qb_1.sample(samples).eval(), qb_2.sample(samples).eval())
for i in range(samples):
	y_hat = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: W0_samp[i, :, :], 
		bb0: b0_samp[i, :], ww1: W1_samp[i, :, :], bb1: b1_samp[i, :],
		ww2: W2_samp[i, :, :], bb2: b2_samp[i, :]})
	plt.plot(x_in, y_hat, color='pink')
# MC estimate (using 500 samples)
y_hat = sess.run(y_pred, feed_dict={x_pred: np.reshape(x_in,[-1, 1]), ww0: W0_opt, bb0: b0_opt, 
	ww1: W1_opt, bb1: b1_opt, ww2: W2_opt, bb2: b2_opt})
plt.plot(x_in, y_hat, color='black') 
plt.scatter(X_test[:, 0], Y_test)
plt.show()