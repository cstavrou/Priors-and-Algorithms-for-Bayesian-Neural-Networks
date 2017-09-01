import numpy as np
from pymc3 import Model, Normal, Laplace, StudentT
import pymc3 as pm
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
import sys
import timeit
import random
# import cv2
import os


# Load the co2 regression data
X_test = np.load('../data/x_test.npy')
X_train = np.load('../data/x_train.npy')
Y_test = np.load('../data/y_test.npy')
Y_train = np.load('../data/y_train.npy')


np.random.seed(seed=314159)

n_hidden = int(sys.argv[1])           # number of hidden units.
N , D = np.shape(X_train)             # number of features.
K = np.shape(Y_test)[1]               # number of classes.
n_samp = int(sys.argv[2])             # number of samples for HMC.
std = float(sys.argv[3])              # prior standard deviation.
std_out = float(sys.argv[4])          # output standard deviation.

# Random initialisation for the weights
W0_init = np.random.laplace(size=[D, n_hidden]).astype(theano.config.floatX)
W1_init = np.random.laplace(size=[n_hidden, K]).astype(theano.config.floatX)
b0_init = np.random.laplace(size=[n_hidden]).astype(theano.config.floatX)
b1_init = np.random.laplace(size=[K]).astype(theano.config.floatX)

# Shared variables
X_shared = theano.shared(X_train)
Y_shared = theano.shared(Y_train)

with Model() as neural_network:

	# Specifying priors
	# =================
	W0 = Laplace('w0', mu=0, b=1, shape=[D, n_hidden],
		testval=W0_init)
	b0 = Laplace('b0', mu=0, b=1, shape=[n_hidden],
		testval=b0_init)
	W1 = Laplace('w1', mu=0, b=1, shape=[n_hidden, K],
		testval=W1_init)
	b1 = Laplace('b1', mu=0, b=1, shape=[K],
		testval=b1_init)

	# Building NN likelihood
	h1 = tt.nnet.softplus(tt.dot(X_shared, W0) + b0)
	mu_est = tt.dot(h1, W1) + b1

	# Regression likelihood
	Normal('y_hat', mu=mu_est, sd=std_out, observed=Y_shared)


# Inference
with neural_network:
	# Sample from posterior
	v_params = pm.advi(n=100000)
	trace = pm.sample_vp(v_params, draws=5000)

print(pm.df_summary(trace))
pm.traceplot(trace)

# Posterior predictive samples
ppc = pm.sample_ppc(trace, samples=500)

pred = ppc['y_hat']
mse = np.mean((pred - Y_train)**2)
print('MC test MSE: ', mse)