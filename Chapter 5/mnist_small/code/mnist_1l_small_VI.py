# =======================================================================================================
# Using the modified (small) MNIST dataset
# =======================================================================================================


# ============
# Description
# ============
# Performs VI on a 1 hidden layer Bayesian NN


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

# Load the (small) MNIST data
X_test = np.load('../data/x_test.npy')
Y_test = np.load('../data/y_test.npy')
X_train = np.load('../data/x_train.npy')
Y_train = np.load('../data/y_train.npy')

D = int(14**2)
ed.set_seed(314159)
np.random.seed(seed=314159)


K = np.shape(Y_train)[1]                # number of classes.

n_hidden = int(sys.argv[1])
inf_iter = int(sys.argv[2]) 		    # number of iterations of VI.
std = float(sys.argv[3])
non_zero_mean = str(sys.argv[4])        # True/False (if mean of posterior distributions is 0)
if str(sys.argv[5]) == 'T':
	df = float(sys.argv[6])

Y_train = np.argmax(Y_train,axis=1)
Y_test = np.argmax(Y_test,axis=1)

# Halves the size of the images
def resize(images):
	im = np.reshape(images, [-1,28,28])
	n = np.shape(im)[0]
	reduced_im = np.zeros([n, 14, 14])
	for ind in range(n):
		reduced_im[ind,:,:] = cv2.resize(im[ind,:,:], (14, 14))
	grey_im = (0.1 < reduced_im).astype('float32')
	return np.reshape(grey_im, [-1, 14*14])

def nn(x, W_0, b_0, W_1, b_1):
    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    out = tf.matmul(h, W_1) + b_1
    return out

def pred_nn(x, W_0, b_0, W_1, b_1):
    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    o = tf.nn.softmax(tf.matmul(h, W_1) + b_1)
    return tf.reshape(tf.argmax(o, 1), [-1])

def probs_nn(x, W_0, b_0, W_1, b_1):
    h = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
    out_probs = tf.nn.softmax(tf.matmul(h, W_1) + b_1)
    return tf.reshape(out_probs, [-1, K])

def mean_acc(Y_true, Y_hat):
    acc = Y_true == Y_hat
    return np.mean(acc)  

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

x = tf.placeholder(tf.float32, [None, None])
y = Categorical(logits=nn(x, W_0, b_0, W_1, b_1))
# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [None])

# Build predictive graph
#
x_pred = tf.placeholder(tf.float32, [None, None])
ww0 = tf.placeholder(tf.float32, [None, None])
ww1 = tf.placeholder(tf.float32, [None, None])
bb0 = tf.placeholder(tf.float32, [None])
bb1 = tf.placeholder(tf.float32, [None])
y_pred = pred_nn(x_pred, ww0, bb0, ww1, bb1)
prob_out = probs_nn(x_pred, ww0, bb0, ww1, bb1)

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

if str(sys.argv[5]) != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter) + 'rep/kl_' + name_add + '/' +  str(sys.argv[5]))
else:
	path =  ('../saved/' + str(n_hidden) +'units/1l_' + str(inference.n_iter) + 
		'rep/kl_' + name_add + '/T_' + str(df).replace('.','_'))

if not os.path.exists(path):
  os.makedirs(path)

# Training
test_acc = []
mini_samp = 100

for _ in range(inference.n_iter):
	# Start timer - make sure only the actual inference part is calculated
	if _ == 0:
		total = timeit.default_timer()
	start = timeit.default_timer()
	info_dict = inference.update(feed_dict={x: resize(X_train), y_ph: Y_train})
	inference.print_progress(info_dict)
	elapsed = timeit.default_timer() - start
	total = total + elapsed
	if (_ + 1 ) % 50 == 0 or _ == 0:
		y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})
		acc_tmp = ed.evaluate('sparse_categorical_accuracy', data={x: resize(X_test), y_post: Y_test}, n_samples=mini_samp)
		print('\nIter ', _+1, ' -- Accuracy: ', acc_tmp)
		test_acc.append(acc_tmp)		

# Save test accuracy during training
name = path + '/test_acc.csv'
np.savetxt(name, test_acc, fmt = '%.5f', delimiter=',')

## Model Evaluation
#
y_post = ed.copy(y, {W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1})


print("Accuracy on test data:")
acc1 = ed.evaluate('sparse_categorical_accuracy', data={x: resize(X_test), y_post: Y_test}, n_samples=mini_samp)
# acc1 = ed.evaluate('sparse_categorical_accuracy', data={x: X_test, y_post: Y_test})
print(acc1)


pred_acc_list = np.zeros([mini_samp])

pW_0, pW_1, pb_0, pb_1 = (qW_0.sample(mini_samp).eval(), qW_1.sample(mini_samp).eval(),
	qb_0.sample(mini_samp).eval(), qb_1.sample(mini_samp).eval())

for i in range(mini_samp):
	pred = sess.run(y_pred, feed_dict={x_pred: resize(X_test), ww0: pW_0[i, :, :], 
		bb0: pb_0[i, :], ww1: pW_1[i, :, :], bb1: pb_1[i, :]})
	acc_tmp = mean_acc(Y_test, pred)
	pred_acc_list[i] = acc_tmp

mc_pred = sess.run(y_pred, feed_dict={x_pred: resize(X_test), ww0: pW_0.mean(axis=0),
	bb0: pb_0.mean(axis=0), ww1: pW_1.mean(axis=0), bb1: pb_1.mean(axis=0)})
mc_acc = mean_acc(Y_test, mc_pred)



rnd0_i = random.sample(range(D), 3)
rnd0_j = random.sample(range(n_hidden), 4)
rnd1_i = random.sample(range(K), 3)
rnd1_j = random.sample(range(n_hidden), 4)
	
# Marginal distribution plots
#	
fig, ax = plt.subplots(2)
for i in range(3):
	for j in range(4):
		sns.distplot(qW_0.sample(500).eval()[:, rnd0_i[i], rnd0_j[j]], hist=False, rug=False, ax=ax[0])
		sns.distplot(qW_1.sample(500).eval()[:, rnd1_j[j], rnd1_i[i]], hist=False, rug=False, ax=ax[1])
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.7)
plt.savefig(path + '/marginal_distribution_plots.png')
plt.close(fig)


# Final prediction using 500 samples
#
qW0_smp, qW1_smp = qW_0.sample(500).eval(), qW_1.sample(500).eval()
qb0_smp, qb1_smp = qb_0.sample(500).eval(), qb_1.sample(500).eval()

acc_final = []
conf_mat = np.zeros([np.shape(qW0_smp)[0], K, K])
probs = np.zeros([np.shape(Y_test)[0], K])

for i in range(np.shape(qW0_smp)[0]):
	pred = sess.run(y_pred, feed_dict={x_pred: resize(X_test), ww0: qW0_smp[i, :, :],
			bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})
	acc_final.append(mean_acc(Y_test, pred))
	tmp_conf = confusion_matrix(Y_test, pred)
	# Express confussion matrix as probabilities
	conf_mat[i, :, :] = (tmp_conf.T/np.sum(tmp_conf, axis=1)).T
	probs = probs + sess.run(prob_out, feed_dict={x_pred: resize(X_test), ww0: qW0_smp[i, :, :],
			bb0: qb0_smp[i, :], ww1: qW1_smp[i, :, :], bb1: qb1_smp[i, :]})

y_hat = np.reshape(np.argmax(probs, axis=1), [-1])
fin_acc = mean_acc(Y_test, y_hat)
print('Final prediction accuracy = ', fin_acc, ' +/- ', str(np.std(acc_final)))

# Save info file
print('Total time elapsed (seconds): ',total)

info = ['Total algorithm time (seconds) -- ' + str(total), 
'Test accuracy (posterior) -- ' + str(acc1), 
'Mean prediction accuracy (100 samples from posterior) -- ' + str(mc_acc),
'Std of prediction accuracy (100 samples from posterior) -- ' + str(np.std(pred_acc_list)),
'Final MC prediction accuracy -- ' + str(fin_acc) + ' +/- ' + str(np.std(acc_final)),
'Prior standard deviation -- ' + str(std)]
name = path + '/info_file.csv'
np.savetxt(name, info, fmt='%s' , delimiter=',')

# Graph of drawn confussion matrices from the posterior
conf_mat_mean = confusion_matrix(Y_test, y_hat)

fig, ax = plt.subplots(1)
cbar = ax.imshow((conf_mat_mean.T/np.sum(conf_mat_mean, axis=1)).T, cmap=plt.cm.gnuplot, interpolation='none')
ax.set_xticks(np.arange(0, 9, 2))
ax.grid(False)
fig.colorbar(cbar, orientation='vertical')
plt.savefig(path + '/predictive_conf_matrix.png')
plt.close(fig)

# Plot the standard deviation of the confussion matrix
fig, ax = plt.subplots(1)
cbar = ax.imshow(np.round(np.std(conf_mat, axis=0), decimals=4), cmap=plt.cm.gnuplot, interpolation='none')
ax.set_xticks(np.arange(0, 9, 2))
ax.grid(False)
fig.colorbar(cbar, orientation='vertical')
plt.subplots_adjust(hspace=0.2)
plt.savefig(path + '/predictive_conf_matrix_std.png')
plt.close(fig)