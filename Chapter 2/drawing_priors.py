# Drawing from different priors
#


from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Inputs
#
# 1.  tanh/relu/softmax           (activation function)
# 2.  T/laplace/normal/cauchy/mix (prior function)
# 3.  n_hidden                    (number of hidden units)
# 4.  df                          (parameter of T-dist)
# 5.  samp                        (number of samples of x1, x2)
# 6.  lim                         (x and y limits for graph)
# 7.  n_layers                    (number of hidden layers)
# 8.  std                         (variance of the prior distributions)
# 9.  seed                        (random seed)
# 10. n_iter                      (number of iterations for resampling prior)
# 11. prior1                      (first prior distribution)  [ONLY if prior = mix]  
# 12. prior2                      (second prior distribution) [ONLY if prior = mix]


prior = str(sys.argv[2])
n_hidden = int(sys.argv[3])
df = float(sys.argv[4])
D = 2    # number of features.
samp = int(sys.argv[5])
lim = float(sys.argv[6])
n_layers = int(sys.argv[7])  
std = float(sys.argv[8])  
seed = int(sys.argv[9])
n_iter = int(sys.argv[10])

# Set the random seed for comparisons
#

# Define the neural network
#
if n_layers == 2:
	def draw_nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
		if str(sys.argv[1]) == 'relu':
			h1 = tf.nn.relu_layer(x, W_0, b_0)
			h2 = tf.nn.relu_layer(h1, W_1, b_1)
		if str(sys.argv[1]) == 'tanh':
			h1 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
			h2 = tf.nn.tanh(tf.matmul(h1, W_1) + b_1)
		if str(sys.argv[1]) == 'softplus':
			h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
			h2 = tf.nn.softplus(tf.matmul(h1, W_1) + b_1)
		y = tf.matmul(h2, W_2) + b_2
		return y

if n_layers == 1:
	def draw_nn(x, W_0, b_0, W_1, b_1):
		if str(sys.argv[1]) == 'relu':
			h1 = tf.nn.relu_layer(x, W_0, b_0)
		if str(sys.argv[1]) == 'tanh':
			h1 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
		if str(sys.argv[1]) == 'softplus':
			h1 = tf.nn.softplus(tf.matmul(x, W_0) + b_0)
		y = tf.matmul(h1, W_1) + b_1
		return y

if n_layers == 0:
	def draw_nn(x, W_0, b_0):
		y = tf.matmul(x, W_0) + b_0
		return y

# Create the meshgrid
#

x1 = np.linspace(-lim, lim, samp)
x2 = np.linspace(-lim, lim, samp)
X, Y = np.meshgrid(x1, x2)

x1 = np.reshape([np.linspace(-lim, lim, samp)]*samp, [-1], order='F')
x2 = np.reshape([np.linspace(-lim, lim, samp)]*samp, [-1])
x_in = np.transpose([x1,x2])



def initialise_par(size, fun, df):
	size_b = [size[len(size) - 1]]
	if fun == 'T':
		weights = tf.cast(np.random.standard_t(df, size=size), tf.float32)
		bias = tf.cast(np.random.standard_t(df, size=size_b), tf.float32)
	if fun == 'cauchy':
		weights = tf.cast(np.random.standard_cauchy(size=size), tf.float32)
		bias = tf.cast(np.random.standard_cauchy(size=size_b), tf.float32)
	if fun == 'laplace':
		weights = tf.cast(np.random.laplace(size=size, scale = std**2/size[0]), tf.float32)
		bias = tf.cast(np.random.laplace(size=size_b, scale = std**2/size[0]), tf.float32)
		# weights = tf.cast(np.random.laplace(size=size, scale = std/size[0]**.5), tf.float32)
		# bias = tf.cast(np.random.laplace(size=size_b, scale = std/size[0]**.5), tf.float32)
	if fun == 'normal':
		weights = tf.cast(np.random.normal(size=size, scale = std/(size[0])**.5), tf.float32)
		bias = tf.cast(np.random.normal(size=size_b, scale = std/(size[0])**.5), tf.float32)
	return tf.Variable(weights), tf.Variable(bias)

if n_layers == 0:
	W_0, b_0 = initialise_par([D, 1], prior, df)
if n_layers == 1:
	W_0, b_0 = initialise_par([D, n_hidden], prior, df)
	W_1, b_1 = initialise_par([n_hidden, 1], prior, df)	
if n_layers == 2 and prior != 'mix':
	W_0, b_0 = initialise_par([D, n_hidden], prior, df)
	W_1, b_1 = initialise_par([n_hidden, n_hidden], prior, df)
	W_2, b_2 = initialise_par([n_hidden, 1], prior, df)
if prior == 'mix':
	prior1 = str(sys.argv[11])
	prior2 = str(sys.argv[12])	
	W_0, b_0 = initialise_par([D, n_hidden], prior1, df)
	W_1, b_1 = initialise_par([n_hidden, n_hidden], prior1, df)
	W_2, b_2 = initialise_par([n_hidden, 1], prior2, df)



x = tf.placeholder(tf.float32, [None, None])

if n_layers == 2:
	y_ = draw_nn(x, W_0, b_0, W_1, b_1, W_2, b_2)

if n_layers == 1:
	y_ = draw_nn(x, W_0, b_0, W_1, b_1)

if n_layers == 0:
	y_ = draw_nn(x, W_0, b_0)


with tf.device('/cpu:0'):
	for i in range(n_iter):
		if n_layers == 2:
			np.random.seed(seed+i)
		with tf.device('/gpu:0'):
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				y_i = sess.run(y_, feed_dict={x: x_in})	
		print(i)
		if i == 0:
			y_new = y_i
		if i > 0:
			y_new = y_new + y_i

y = y_new/n_iter


Z = np.reshape(y, [samp, samp])

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1, projection='3d')

# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
#ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=1, cmap=cm.CMRmap)
# cset = ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=-1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=4, cmap=cm.coolwarm)
plt.show()