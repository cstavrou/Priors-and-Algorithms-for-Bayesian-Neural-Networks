import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import seaborn as sns

n_hidden = str(sys.argv[1])
mod = str(sys.argv[2])
prior = str(sys.argv[3])
n_reps = str(sys.argv[4])
non_zero_mean = str(sys.argv[5])
start = int(sys.argv[6])

if non_zero_mean == 'True':
	method = 'kl_non_zero_mean'
else:
	method = 'kl'

path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_reps) + 'rep/' + 
	method + '/' + prior)
if prior != 'T':
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_reps) + 
		'rep/' + method + '/' + prior)
else:
	path =  ('../saved/' + str(n_hidden) +'units/' + str(mod) + '_' + str(n_reps) + 
		'rep/' + method + '/' + prior + '_' + str(df).replace('.','_'))

acc = pd.read_csv(path + '/test_acc.csv', header=None)[start:]

fig, ax = plt.subplots(1)
it = 50*np.arange(len(acc)) + 50*start
ax.plot(it, acc)
ax.set_xlabel('Iteration', fontsize=14) 
ax.set_ylabel('Accuracy', fontsize=14)
if start > 0:
	plt.savefig(path + '/acc_plot_skip_first.png')
else:
	plt.savefig(path + '/acc_plot.png')
plt.close(fig)